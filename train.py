import math
import logging
import time
import sys
import os
import argparse
import torch
import numpy as np
import random
from pathlib import Path
from evaluation.evaluation import eval_edge_prediction
from model.tgn_model import TGN
from utils.util import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.cache import get_cache_plan
from utils.data_processing import get_data
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

parser = argparse.ArgumentParser('TGN self-supervised training with embedding reuse')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')
parser.add_argument('--enable_random', action='store_true',help='use random seeds')
parser.add_argument('--budget', type=int, default=0, help='budget on the number of cached nodes')
parser.add_argument('--gradient', action='store_true',help='enable history with gradients')
parser.add_argument('--log', action='store_true',help='log data distribution')
parser.add_argument('--time', type=str, default="real", help='[real|order]')
parser.add_argument('--clip_value', action='store_true',help='clip gradient value')
parser.add_argument('--clip_norm', action='store_true',help='clip gradient norm')
parser.add_argument('--clip', type=float, default=1, help='the gradient clipping value')
parser.add_argument('--reuse', action='store_true', help='reuse historical embeddings')
parser.add_argument('--reuse_test', action='store_true',help='reuse when testing')
parser.add_argument('--cache_strategy', type=str, default="MRU", help='[MRU|LRU|2Q]')


args = parser.parse_args()
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
BATCH_SIZE = args.bs

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'

cur_time=time.time()
best_checkpoint_path = f'./saved_checkpoints/{args.data}-{args.n_epoch}-{args.lr}-{args.reuse}-{time.time()}.pth'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if not args.enable_random:
  torch.manual_seed(0)
  np.random.seed(0)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

### get file name
filename=args.data
if args.reuse:           
  filename=filename+'_train_reuse'
  if args.reuse_test:    
    filename=filename+'_test_reuse'
  if args.budget!=0:     
    filename=filename+'_budget_'+str(args.budget)+'_strategy_'+args.cache_strategy
  if args.gradient: 
    filename=filename+'_gradient'
filename=filename+'_bs_'+str(BATCH_SIZE)+'_'+args.aggregator+'_layer_'+str(args.n_layer)+'_epoch_'+str(args.n_epoch)+'_lr_'+str(args.lr)
if args.enable_random:
  filename=filename+'_random_seed'
if args.clip_norm:
  filename=filename+'_clip_norm_'+str(args.clip)
if args.clip_value:
  filename=filename+'_clip_value_'+str(args.clip)
print(filename)


### get logger
Path(f"log/{args.data}").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(f'log/{args.data}/{filename}')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


### get dataset and sampler 
node_features, edge_features, full_data, full_train_data, full_val_data, test_data, new_node_val_data,new_node_test_data = get_data(DATA)
train_ngh_finder = get_neighbor_finder(full_train_data)
full_ngh_finder = get_neighbor_finder(full_data)

### negative sampler
train_rand_sampler = RandEdgeSampler(full_train_data.sources, full_train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,new_node_test_data.destinations,seed=3)

device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

############# get cache plan #############
if args.reuse and args.budget>0:
  strategy = args.cache_strategy
  num_embeddings=node_features.shape[0]
  t_start=time.time()
  train_cache_plan_list=get_cache_plan(args,full_train_data,train_ngh_finder,num_embeddings,strategy)
  t_cache = time.time()-t_start
  print(f'cache time {t_cache}')


for i in range(args.n_runs):
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            embedding_module_type=args.embedding_module, 
            message_function=args.message_function, 
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            args=args)

  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  t_total_epoch_train=0
  t_total_epoch_val=0
  stop_epoch=-1


  for epoch in range(NUM_EPOCH):
    t_epoch_train_start = time.time()
    train_data = full_train_data
    val_data = full_val_data
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance/BATCH_SIZE)

    train_ap=[]
    train_acc=[]
    train_auc=[]
    train_loss=[]

    tgn.memory.__init_memory__()
    if args.reuse:
      tgn.embedding_module.reset_history()
    tgn.set_neighbor_finder(train_ngh_finder)


    for batch_idx in tqdm(range(0, num_batch)):
      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      sample_inds = np.array(list(range(start_idx,end_idx)))

      sources_batch, destinations_batch = train_data.sources[sample_inds],train_data.destinations[sample_inds]
      edge_idxs_batch = train_data.edge_idxs[sample_inds]
      timestamps_batch = train_data.timestamps[sample_inds]
      size = len(sources_batch)
      _, negatives_batch = train_rand_sampler.sample(size)

      with torch.no_grad():
        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)

      cache_plan=train_cache_plan_list[batch_idx] if args.reuse and args.budget!=0 else None
      tgn = tgn.train()
      optimizer.zero_grad()
      pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS, reuse=args.reuse, train=True, cache_plan=cache_plan)
      loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
      loss.backward()
      train_loss.append(loss.item())

      with torch.no_grad():
        pos_prob=pos_prob.cpu().numpy() 
        neg_prob=neg_prob.cpu().numpy()
        pred_score = np.concatenate([pos_prob, neg_prob]) 
        true_label = np.concatenate([np.ones(size), np.zeros(size)])
        true_binary_label= np.zeros(size)
        pred_binary_label = np.argmax(np.hstack([pos_prob,neg_prob]),axis=1)
        train_ap.append(average_precision_score(true_label, pred_score))
        train_auc.append(roc_auc_score(true_label, pred_score))
        train_acc.append(accuracy_score(true_binary_label, pred_binary_label))

      # show the effect of gradient clipping
      if args.clip_norm or args.clip_value:
        print_clip_before=False
        print_clip_after=False

        if print_clip_before:
          # torch.norm calculates the Frobenius norm
          norm_before=torch.norm(torch.cat([p.grad.view(-1) for p in tgn.embedding_module.attention_models.parameters()]))
          max_before=torch.max(torch.abs(torch.cat([p.grad.view(-1) for p in tgn.embedding_module.attention_models.parameters()])))
          print(f'[before] norm {norm_before}, max {max_before}')
    
        if args.clip_norm:
          torch.nn.utils.clip_grad_norm_(tgn.embedding_module.attention_models.parameters(),args.clip)
        elif args.clip_value:
          torch.nn.utils.clip_grad_value_(tgn.embedding_module.attention_models.parameters(),args.clip)
      
        if print_clip_after:
          norm_after=torch.norm(torch.cat([p.grad.view(-1) for p in tgn.embedding_module.attention_models.parameters()]))
          max_after=torch.max(torch.abs(torch.cat([p.grad.view(-1) for p in tgn.embedding_module.attention_models.parameters()])))
          print(f'[after] norm {norm_after}, max {max_after}')

      optimizer.step()

      if args.reuse and args.gradient:
        tgn.embedding_module.detach_history()

    epoch_train_time = time.time() - t_epoch_train_start
    t_total_epoch_train+=epoch_train_time
    train_ap=np.mean(train_ap)
    train_auc=np.mean(train_auc)
    train_acc=np.mean(train_acc)
    train_loss=np.mean(train_loss)

    ######################## Model Validation on the Val Dataset #######################
    t_epoch_val_start=time.time()

    ### transductive val
    tgn.set_neighbor_finder(full_ngh_finder)
    train_memory_backup = tgn.memory.backup_memory()
    if args.reuse and args.reuse_test:
      train_history_backup = tgn.embedding_module.backup_history()

    val_ap, val_auc, val_acc = eval_edge_prediction(model=tgn,negative_edge_sampler=val_rand_sampler,data=val_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE, reuse=args.reuse and args.reuse_test,cache_plan=None)

    val_memory_backup = tgn.memory.backup_memory()
    if args.reuse and args.reuse_test:
      val_history_backup = tgn.embedding_module.backup_history()
    tgn.memory.restore_memory(train_memory_backup)
    if args.reuse and args.reuse_test:
      tgn.embedding_module.restore_history(train_history_backup)

    ### inductive val
    nn_val_ap, nn_val_auc, nn_val_acc = eval_edge_prediction(model=tgn,negative_edge_sampler=val_rand_sampler,data=new_node_val_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE,reuse=args.reuse and args.reuse_test,cache_plan=None)
    tgn.memory.restore_memory(val_memory_backup)
    if args.reuse and args.reuse_test:
      tgn.embedding_module.restore_history(val_history_backup)

    epoch_val_time = time.time() - t_epoch_val_start
    t_total_epoch_val += epoch_val_time





    ########## logging val performance ##########
    epoch_id = epoch+1
    logger.info('epoch: {}, train: {}, val: {}'.format(epoch_id, epoch_train_time,epoch_val_time))
    logger.info('train auc: {}, train ap: {}, train acc: {}, train loss: {}'.format(train_auc,train_ap,train_acc,train_loss))
    logger.info('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))
    logger.info('val acc: {}, new node val acc: {}'.format(val_acc, nn_val_acc))


    ########## early stopping check #############
    last_best_epoch=early_stopper.best_epoch
    
    if early_stopper.early_stop_check(val_ap):
      stop_epoch=epoch_id
      model_parameters,tgn.memory=torch.load(best_checkpoint_path)
      tgn.load_state_dict(model_parameters)
      tgn.eval()
      break
    else:
      if epoch==early_stopper.best_epoch:
        torch.save((tgn.state_dict(),tgn.memory), best_checkpoint_path)


  ###################### Evaludate Model on the Test Dataset #######################
  t_test_start=time.time()

  ### transductive test
  val_memory_backup = tgn.memory.backup_memory()
  if args.reuse and args.reuse_test:
    val_history_backup = tgn.embedding_module.backup_history()

  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc, test_acc = eval_edge_prediction(model=tgn,negative_edge_sampler=test_rand_sampler,data=test_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE,reuse=args.reuse and args.reuse_test,cache_plan=None)

  tgn.memory.restore_memory(val_memory_backup)
  if args.reuse and args.reuse_test:
    tgn.embedding_module.restore_history(val_history_backup)

  ### inductive test
  nn_test_ap, nn_test_auc, nn_test_acc = eval_edge_prediction(model=tgn,negative_edge_sampler= nn_test_rand_sampler, data=new_node_test_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE, reuse=args.reuse and args.reuse_test,cache_plan=None)
  t_test=time.time()-t_test_start

  NUM_EPOCH=stop_epoch if stop_epoch!=-1 else NUM_EPOCH
  logger.info(f'### num_epoch {NUM_EPOCH}, epoch_train {t_total_epoch_train/NUM_EPOCH}, epoch_val {t_total_epoch_val/NUM_EPOCH}, epoch_test {t_test}')
  logger.info('Test statistics: Old nodes -- auc: {}, ap: {}, acc: {}'.format(test_auc, test_ap, test_acc))
  logger.info('Test statistics: New nodes -- auc: {}, ap: {}, acc: {}'.format(nn_test_auc, nn_test_ap, nn_test_acc))
  os.remove(best_checkpoint_path)
