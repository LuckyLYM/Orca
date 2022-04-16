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
from evaluation.evaluation import eval_edge_prediction, eval_edge_prediction_tbatch
from model.tgn_model import TGN
from utils.util import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, get_cache_plan_FIF_multiple_stages,get_cache_plan_FIF
from utils.data_processing import get_data, compute_time_statistics,merge_last_two
from tqdm import tqdm

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
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', default=True, type=bool, help='Whether to augment the model with a node memory')
parser.add_argument('--memory_update_at_end', action='store_true', help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--different_new_nodes', action='store_true',help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',help='Whether to run the dyrep model')

# --------- check the individual components here
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')

# --------- some model specific new variables
parser.add_argument('--enable_random', action='store_true',help='use random seeds')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--ratio', type=float, default=1.0, help='sampling ratio')
parser.add_argument('--reuse_test', action='store_true',help='reuse when testing')
parser.add_argument('--topk', type=int, default=0, help='compute the exact embedding for top-k out-of-batch neighbors')
parser.add_argument('--budget', type=int, default=0, help='budget on the number of cached nodes, -1 indicates recomputing all the 1-hop out-of-batch neighbors, 0 indicates reuse all')
parser.add_argument('--gradient', action='store_true',help='enable history with gradients')
parser.add_argument('--log', action='store_true',help='log data distribution')
parser.add_argument('--accurate', action='store_true',help='compute more nodes from better accuracy')
parser.add_argument('--rule',type=str, default="stale", help='[count|stale|mix]')
parser.add_argument('--unique', action='store_true',help='aggregate information from unique neighbors')
parser.add_argument('--max_time', action='store_true',help='maximum time for embedding')
parser.add_argument('--mean_time', action='store_true',help='mean time for embedding calculation')
parser.add_argument('--min_time', action='store_true',help='mean time for embedding calculation')
parser.add_argument('--time', type=str, default="real", help='[real|order]')
parser.add_argument('--clip_value', action='store_true',help='clip gradient value')
parser.add_argument('--clip_norm', action='store_true',help='clip gradient norm')
parser.add_argument('--clip', type=float, default=1, help='the gradient clipping value')

# --------- check data loader, and feature dimension here etc...
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for each user')
parser.add_argument('--diverse_time', action='store_true',help='mean time for embedding calculation')

# --------- for temporal graph partition
parser.add_argument('--sampling_strategy', type=str, default="epoch", help='[epoch|batch|partition]')
parser.add_argument('--staleness', type=int, default=30, help='staleness constraint')
parser.add_argument('--sequential', action='store_true',help='adopt the sequential batch version')
parser.add_argument('--maximum_batch_size', type=int, default=-1, help='maximum batch size in graph partition')
parser.add_argument('--partition_test', action='store_true',help='partition the test data')

# TODO: we implement two types of optimization skills here
parser.add_argument('--new', action='store_true', help='using the temmporal embeddings as query')
parser.add_argument('--reuse', action='store_true', help='reuse historical embeddings')
parser.add_argument('--optimize_memory', action='store_true',help='reduce the cost of get updated memory')
parser.add_argument('--immediate_memory', action='store_true',help='reduce the cost of get updated memory')



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
USE_MEMORY = True
NODE_DIM = args.node_dim


TIME_DIM = args.time_dim
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
BATCH_SIZE = args.bs
Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
cur_time=time.time()
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{cur_time}-{args.prefix}-{args.data}-{epoch}.pth'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if not args.enable_random:
  torch.manual_seed(0)
  np.random.seed(0)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


######################## get file name ########################
strategy=args.sampling_strategy
ratio=args.ratio
new_data=args.data
filename=new_data
if args.new:
  filename=filename+'_new' 
if args.unique:
  filename=filename+'_unique'
if args.reuse:            # reuse historical embeddings
  filename=filename+'_train_reuse'
  if args.gradient:       # enable history with gradients
    filename=filename+'_gradient'
  if args.accurate:       # accurate reuse
    supported_rules=['stale','count','mix']
    assert(args.topk!=0)
    assert(args.rule in supported_rules)
    filename=filename+'_accurate_'+args.rule+'_top_'+str(args.topk)
  if args.reuse_test:     # reuse when test model on test data
    filename=filename+'_all'
  if args.budget!=0:      # cache #budget nodes
    filename=filename+'_budget_'+str(args.budget)

if strategy=='partition': # stalness-aware graph partition
  if args.sequential:     # sequential staleness partition
    filename=filename+'_seq_'+str(args.staleness)
  else:                   # temporal staleness partition
    filename=filename+'_one_hop_'+str(args.staleness)
  if args.maximum_batch_size!=-1:
    filename=filename+'_upper_'+str(args.maximum_batch_size)
  if args.partition_test:
    filename=filename+'_partition_test'
else:                     
  filename=filename+'_bs_'+str(BATCH_SIZE)

# sampling strategy + sampling ratio + aggregator + #layers
filename=filename+'_'+args.aggregator+'_layer_'+str(args.n_layer)+'_epoch_'+str(args.n_epoch)+'_lr_'+str(args.lr)
if args.enable_random:
  filename=filename+'_random_seed'
if args.clip_norm:
  filename=filename+'_clip_norm_'+str(args.clip)
if args.clip_value:
  filename=filename+'_clip_value_'+str(args.clip)



if args.optimize_memory:
  filename=filename+'_optimize_memory'
elif args.immediate_memory:
  filename=filename+'_immediate_memory'
print(filename)



######################## get logger ########################
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
#logger.info(args)


######################## get dataset and sampler ########################
t0=time.time()
if strategy=='epoch' or strategy=='batch':
  node_features, edge_features, full_data, full_train_data, full_val_data, test_data, new_node_val_data,new_node_test_data = get_data(DATA)
elif strategy=='partition':
  node_features, edge_features, full_data, full_train_data, full_val_data, test_data, new_node_val_data,new_node_test_data = get_data(DATA,args.staleness,args.n_degree,args.sequential,args.maximum_batch_size)
t_data=time.time()-t0

print(f'--------------- data loading time {t_data} ---------------')
train_ngh_finder = get_neighbor_finder(full_train_data, args.uniform,time_encoding=args.time,unique=args.unique)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform,time_encoding=args.time,unique=args.unique)

# negative sampler
train_rand_sampler = RandEdgeSampler(full_train_data.sources, full_train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,new_node_test_data.destinations,seed=3)

device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)


######################## get cache plan ########################
if args.reuse and args.budget>0:
  print('--------------- get cache plan --------------')
  num_embeddings=node_features.shape[0]
  cache_flag=np.zeros(num_embeddings)

  if False:
    print('--------------- compare FIF and reuse ratio algorithms --------------')
    #budget_list=[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
    #budget_list=[3000,4000,5000,6000,7000,8000,9000,10000]
    
    #budget_list=[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    #budget_list=[3000,4000,5000,6000,7000,8000,9000,10000]
    #budget_list=[9000,10000]
    budget_list=[20000,30000,40000,50000,60000]
    print(budget_list)
    n_reuse_list=[]
    n_recompute_list=[]
    cache_hit_list=[]
    reuse_distance_list = []
    for budget in budget_list:
      args.budget=budget
      num_embeddings=node_features.shape[0]
      cache_flag=np.zeros(num_embeddings)
      train_cache_plan_list,n_reuse,n_recompute,n_in,total_reuse_distance=get_cache_plan_FIF(args,cache_flag,full_train_data,train_ngh_finder,num_embeddings)
      n_reuse_list.append(n_reuse)
      n_recompute_list.append(n_recompute)
      cache_hit_list.append((n_reuse+n_in)/(n_reuse+n_recompute+n_in))
      reuse_distance_list.append(total_reuse_distance)
    print(budget_list)
    print(n_reuse_list)
    print(n_recompute_list)
    print(cache_hit_list)
    print(reuse_distance_list)
    sys.exit(1)


  if True:
    print('--------------- get cache plan for train only --------------')
    train_cache_plan_list,_,_,_,_=get_cache_plan_FIF(args,cache_flag,full_train_data,train_ngh_finder,num_embeddings)
    #sys.exit(1)
  else:
    print('--------------- get cache plan for train, val and test --------------')
    data_list=[full_train_data,full_val_data,test_data]
    cache_plans,n_reuse,n_recompute=get_cache_plan_FIF_multiple_stages(args,cache_flag,data_list,full_ngh_finder,num_embeddings)
    train_cache_plan_list=cache_plans[0]
    val_cache_plan_list=cache_plans[1]
    test_cache_plan_list=cache_plans[2]
    print(f'train_batch {len(train_cache_plan_list)}, val_batch {len(val_cache_plan_list)}, test_batch {len(test_cache_plan_list)}')
  
    

for i in range(args.n_runs):

  print(f'input size: node_features {node_features.shape}, edge_feature {edge_features.shape}, memory {MEMORY_DIM}, message {MESSAGE_DIM}')

  ######################## model initialization ########################
  # during the initialization stage, node and edge features will be transferred to GPU
  # in the current version, we set memory dimension equal to be node feature dimension
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module, 
            message_function=args.message_function, 
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep,
            args=args)

  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)
  
  new_nodes_val_aps = []
  val_aps = []
  epoch_train_times = []
  epoch_val_times = []
  epoch_train_val_times = []
  train_losses = []
  early_stopper = EarlyStopMonitor(max_round=args.patience)
  

  ##### the following values are summed across multiple iterations ####
  t_total_epoch_train=0
  t_total_epoch_val=0
  t_total_epoch_train_val=0
  t_total_data_preparation=0
  t_total_train=0
  t_total_memory_management=0
  t_test=0
  stop_epoch=-1


  ################ enumerate training epochs ###############
  for epoch in range(NUM_EPOCH):
    t_epoch_train_start = time.time()
    tgn.reset_timer()
    ######### data sampling per epoch #########
    if strategy=='epoch' and ratio!=1:
      train_data=full_train_data.sample(ratio)
      val_data=full_val_data.sample(ratio)
      print(f'full_train: {train_data.n_interactions}  sample_train: {train_data.n_interactions}' )
      print(f'full_val: {full_val_data.n_interactions}  sample_val: {val_data.n_interactions}' )
    else:
      train_data = full_train_data
      val_data = full_val_data

    ######## reset memory and history here ########
    tgn.memory.__init_memory__()
    if args.reuse:
      tgn.embedding_module.reset_history()
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []
    #logger.info('start {} epoch'.format(epoch))

    ###### decide # of training batches ######
    num_instance = len(train_data.sources)
    if strategy=='partition':
      num_batch = train_data.n_batch
    else:
      num_batch = math.ceil(num_instance/BATCH_SIZE)


    t_batch_train=0
    t_batch_prepare=0
    t_batch_memory=0
    t_batch_backward=0
    t_batch_forward=0
    ######################## training iterations ########################
    for batch_idx in tqdm(range(0, num_batch)):

      ########## prepare a training batch ##########
      t_data_prepartion_start=time.time()
      # get index of a training batch
      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      sample_inds=np.array(list(range(start_idx,end_idx)))
      # sampling from a batch
      if strategy=='batch':
        batch_size=end_idx-start_idx
        sample_size=int(batch_size*ratio)
        sample_inds=random.sample(range(start_idx,end_idx),sample_size)
      # or sampling from a partition
      elif strategy=='partition':
        batch_inds=train_data.tbatch[batch_idx]
        sample_size=int(len(batch_inds)*ratio)
        sample_inds=random.sample(batch_inds,sample_size)
      # load a batch of training data
      sources_batch, destinations_batch = train_data.sources[sample_inds],train_data.destinations[sample_inds]
      edge_idxs_batch = train_data.edge_idxs[sample_inds]
      timestamps_batch = train_data.timestamps[sample_inds]
      size = len(sources_batch)
      # negative sampling
      _, negatives_batch = train_rand_sampler.sample(size)
      # training data does not require gradients
      with torch.no_grad():
        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)
      t_batch_prepare+=time.time()-t_data_prepartion_start


      ########## model trianing on a batch ##########
      cache_plan=train_cache_plan_list[batch_idx] if args.reuse and args.budget!=0 else None
      t_train_start=time.time()
      tgn = tgn.train()
      optimizer.zero_grad()

      t_forward_start=time.time()
      pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS, reuse=args.reuse,cache_plan=cache_plan)
      t_batch_forward+=time.time()-t_forward_start

      t_backward_start=time.time()
      loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
      loss.backward()

      ########## analyze the effect of gradient clipping ##########
      if args.clip_norm or args.clip_value:
        #print('--------------- analyze the effect of gradient clipping ---------------')
        '''
        norm_before=torch.norm(torch.cat([p.grad.view(-1) for p in tgn.embedding_module.attention_models.parameters()]))
        max_before=torch.max(torch.abs(torch.cat([p.grad.view(-1) for p in tgn.embedding_module.attention_models.parameters()])))
        print(f'[before] norm {norm_before}, max {max_before}')
        '''
        if args.clip_norm:
          torch.nn.utils.clip_grad_norm_(tgn.embedding_module.attention_models.parameters(),args.clip)
        elif args.clip_value:
          torch.nn.utils.clip_grad_value_(tgn.embedding_module.attention_models.parameters(),args.clip)
        '''
        norm_after=torch.norm(torch.cat([p.grad.view(-1) for p in tgn.embedding_module.attention_models.parameters()]))
        max_after=torch.max(torch.abs(torch.cat([p.grad.view(-1) for p in tgn.embedding_module.attention_models.parameters()])))
        print(f'[after] norm {norm_after}, max {max_after}')
        '''
      optimizer.step()
      m_loss.append(loss.item())
      t_batch_backward+=time.time()-t_backward_start
      t_batch_train+=time.time()-t_train_start


      ########## detach memory and history ##########
      t_memory_management_start=time.time()
      tgn.memory.detach_memory()
      if args.reuse and args.gradient:
        tgn.embedding_module.detach_history()
      t_batch_memory+=time.time()-t_memory_management_start
      #print(f'after detach memory {tgn.memory.memory.requires_grad}')
    ################## end of training iterations in an epoch ##################

    # * update timer
    epoch_train_time = time.time() - t_epoch_train_start
    epoch_train_times.append(epoch_train_time)
    t_total_epoch_train+=epoch_train_time
    t_total_data_preparation+=t_batch_prepare
    t_total_train+=t_batch_train
    t_total_memory_management+=t_batch_memory
    t_val_memory=0
    t_val_eval=0


    ################## Time breakdown of training stages ##################
    if True:
      a=1
      #print('[epoch time breakdown] epoch_train: {}, batch_train: {}, batch_forward: {}, batch_backward: {}, batch_prepare: {}, batch_memory: {}'.format(epoch_train_time,t_batch_train,t_batch_forward,t_batch_backward,t_batch_prepare,t_batch_memory))
      
      #logger.info('[epoch time breakdown] epoch_train: {}, batch_train: {}, batch_forward: {}, batch_backward: {}, batch_prepare: {}, batch_memory: {}'.format(epoch_train_time,t_batch_train,t_batch_forward,t_batch_backward,t_batch_prepare,t_batch_memory))
      #print('[train time breakdown] get_message: {}, clear_message: {} store_message: {}, get_memory: {}, update_memory: {}, embedding: {}'.format(tgn.t_get_message,tgn.t_clear_message,tgn.t_store_message,tgn.t_get_memory,tgn.t_update_memory,tgn.t_embedding))

      #print('[embedding time breakdown] time_encoding: {}, neighbor: {}, aggregate: {}, push_and_pull: {}, extra_neighbor: {}'.format(tgn.embedding_module.t_time_encoding,tgn.embedding_module.t_neighbor,tgn.embedding_module.t_aggregate,tgn.embedding_module.t_push_and_pull,tgn.embedding_module.t_extra_neighbor))

      #print('[get memory time breakdown] get_memory: {}, index: {}, real_update: {}, others: {}'.format(tgn.t_get_memory,tgn.memory_updater.t_index,tgn.memory_updater.t_real_update,tgn.memory_updater.t_others))
      
      #print('[neighbor time breakdown] neighbor: {}, find_before: {}, sample: {}'.format(tgn.embedding_module.t_neighbor,tgn.embedding_module.neighbor_finder.t_find_before,tgn.embedding_module.neighbor_finder.t_sample))





    ######################## Model Validation on the Val Dataset #######################
    t_epoch_val_start=time.time()
    tgn.set_neighbor_finder(full_ngh_finder)

    ##### backup memory and history #####
    train_memory_backup = tgn.memory.backup_memory()
    if args.reuse and args.reuse_test:
      train_history_backup = tgn.embedding_module.backup_history()
    t_val_memory+=time.time()-t_epoch_val_start

    ##### full val data #####
    t_val_eval_start=time.time()
    val_ap, val_auc = eval_edge_prediction(model=tgn,negative_edge_sampler=val_rand_sampler,data=val_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE, strategy = strategy, ratio= ratio,reuse=args.reuse and args.reuse_test,cache_plan=None,use_partition=True)
    t_val_eval+=time.time()-t_val_eval_start

    ##### backup memory #####
    t_memory_start=time.time()
    val_memory_backup = tgn.memory.backup_memory()
    if args.reuse and args.reuse_test:
      val_history_backup = tgn.embedding_module.backup_history()

    ##### restore memory and history #####
    tgn.memory.restore_memory(train_memory_backup)
    if args.reuse and args.reuse_test:
      tgn.embedding_module.restore_history(train_history_backup)
    t_val_memory+=time.time()-t_memory_start

    ##### new val data #####
    t_val_eval_start=time.time()
    nn_val_ap, nn_val_auc = eval_edge_prediction(model=tgn,negative_edge_sampler=val_rand_sampler,data=new_node_val_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE,strategy = strategy, ratio= ratio, reuse=args.reuse and args.reuse_test,cache_plan=None,use_partition=True)
    t_val_eval+=time.time()-t_val_eval_start

    ##### restore memory and history #####
    t_memory_start=time.time()
    tgn.memory.restore_memory(val_memory_backup)
    if args.reuse and args.reuse_test:
      tgn.embedding_module.restore_history(val_history_backup)
    t_val_memory+=time.time()-t_memory_start


    ##### reocrd val performance #####
    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # * update timer
    epoch_val_time = time.time() - t_epoch_val_start
    epoch_val_times.append(epoch_val_time)
    epoch_train_val_time = time.time()-t_epoch_train_start
    epoch_train_val_times.append(epoch_train_val_time)
    t_total_epoch_val += epoch_val_time
    t_total_epoch_train_val += epoch_train_val_time


    ########## logging val performance ##########
    logger.info('epoch: {}, epoch_train: {}, batch_train: {}, batch_prepare: {}, batch_memory: {}'.format(epoch, epoch_train_time,t_batch_train,t_batch_prepare,t_batch_memory))
    #logger.info('epoch: {}, epoch_val: {}, memory: {}, evaluation: {}'.format(epoch, epoch_val_time,t_val_memory,t_val_eval))
    #logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info('val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))


    ########## early stopping check #############
    #### there is an epoch counter in early_stopper
    #### it is self incremented after each invoke
    last_best_epoch=early_stopper.best_epoch
    if early_stopper.early_stop_check(val_ap):
      #logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      stop_epoch=epoch+1
      #logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      ##### load the best ever model here ####
      model_parameters,tgn.memory=torch.load(best_model_path)
      tgn.load_state_dict(model_parameters)
      #logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      #### we only store the best possible model
      if epoch==early_stopper.best_epoch:
        print(f'----------------- save model {epoch} -----------------')
        torch.save((tgn.state_dict(),tgn.memory), get_checkpoint_path(epoch))
        #### remove last best model
        if epoch!=0:
          path=get_checkpoint_path(last_best_epoch)
          os.remove(path)
          print(f'----------------- remove model {last_best_epoch} -----------------')

  ############### finish all epochs of training and val ############





  ######## logging epoch time ########
  NUM_EPOCH=stop_epoch if stop_epoch!=-1 else NUM_EPOCH
  logger.info(f'### num_epoch {NUM_EPOCH}, epoch_train {t_total_epoch_train/NUM_EPOCH}, epoch_val {t_total_epoch_val/NUM_EPOCH}, epoch_train_val {t_total_epoch_train_val/NUM_EPOCH}, batch_train {t_total_train/NUM_EPOCH} , batch_data_prepare {t_total_data_preparation/NUM_EPOCH}, batch_memory_management {t_total_memory_management/NUM_EPOCH}')


  ###################### Evaludate Model on the Test Dataset #######################
  t_test_start=time.time()
  ##### backup memory #####
  val_memory_backup = tgn.memory.backup_memory()
  if args.reuse and args.reuse_test:
    val_history_backup = tgn.embedding_module.backup_history()

  ##### full test data #####
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc = eval_edge_prediction(model=tgn,negative_edge_sampler=test_rand_sampler,data=test_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE,strategy = strategy, ratio= ratio, reuse=args.reuse and args.reuse_test,cache_plan=None,use_partition=args.partition_test)

  ##### restore memory #####
  tgn.memory.restore_memory(val_memory_backup)
  if args.reuse and args.reuse_test:
    tgn.embedding_module.restore_history(val_history_backup)

  ##### new test data #####
  nn_test_ap, nn_test_auc = eval_edge_prediction(model=tgn,negative_edge_sampler= nn_test_rand_sampler, data=new_node_test_data,n_neighbors=NUM_NEIGHBORS,batch_size=BATCH_SIZE,strategy = strategy, ratio= ratio, reuse=args.reuse and args.reuse_test,cache_plan=None,use_partition=args.partition_test)
  t_test=time.time()-t_test_start

  ##### final logging #####
  logger.info('Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  logger.info('Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))


  #### remove last best model
  print(f'----------------- remove model {early_stopper.best_epoch} -----------------')
  best_model_path = get_checkpoint_path(early_stopper.best_epoch)
  os.remove(best_model_path)

