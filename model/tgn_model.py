import logging
import numpy as np
import torch
import time
from collections import defaultdict
from utils.util import MergeLayer
from modules.memory import Memory, Immediate_Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp", # identity of mlp
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False,
               args=None):
    super(TGN, self).__init__()

    self.t_get_message=0
    self.t_store_message=0
    self.t_embedding=0
    self.t_get_memory=0
    self.t_update_memory=0


    self.batch_counter=0
    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)
    self.args=args

    # TODO: we'd better fix node_features before releasing the code

    # edge features are used in building raw messages, and graph attention module
    # node featuers are used in building temporal embedding, they sum memory and node features
    # (data transfer) first move all the raw features to device (GPU)
    #self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
    #self.n_nodes = self.node_raw_features.shape[0]
    self.n_nodes = node_features.shape[0]

    # 0.7 GB for large_wiki
    #print(f'node feature size {self.node_raw_features.element_size() * self.node_raw_features.nelement()/1024/1024/1024} GB')
    #print(f'edge feature size {self.edge_raw_features.element_size() * self.edge_raw_features.nelement()/1024/1024/1024} GB')


    # dimension of node and edge features
    #self.n_node_features = self.node_raw_features.shape[1]
    self.n_node_features = node_features.shape[1]
    self.n_edge_features = self.edge_raw_features.shape[1]


    # * set dimension of the intermediate embeddings to node feature size
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type

    # false by default
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    # dimension of time features = diemnsion of node features
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.use_memory = use_memory
    self.memory = None
    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst
    self.memory_update_at_start = memory_update_at_start

    # * memory and node should have the same feature dimension, since they sum the two
    #self.memory_dimension = memory_dimension
    self.memory_dimension=self.n_node_features

    # * recompute the message dimension here
    # 2*MEM_DIM+EDGE_DIM+TIME_DIM
    raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                            self.time_encoder.dimension
    
    # decide the message dimension
    message_dimension = message_dimension if message_function != "identity" else raw_message_dimension


    # TODO: modify this part here...
    # build memory
    if self.args.immediate_memory:
      print('\n------------------------------------------')
      print('----------- immediate memory -------------\n')

      self.memory = Immediate_Memory(n_nodes=self.n_nodes,
                          memory_dimension=self.memory_dimension,
                          input_dimension=message_dimension,
                          message_dimension=message_dimension,
                          device=device)

    else:
      self.memory = Memory(n_nodes=self.n_nodes,
                          memory_dimension=self.memory_dimension,
                          input_dimension=message_dimension,
                          message_dimension=message_dimension,
                          device=device)


    # (data transfer) all the following models are on GPU
    # default aggregator is last
    self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,device=device)

    # default message function is identity
    self.message_function = get_message_function(module_type=message_function,
                                                  raw_message_dimension=raw_message_dimension,
                                                  message_dimension=message_dimension)
    
    # default updater is GRU
    self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                              message_dimension=message_dimension,
                                              memory_dimension=self.memory_dimension,
                                              device=device,immediate_memory=self.args.immediate_memory)


    # tgn.embedding_module.attention_models[0]
    # default embedding_module is graph attention
    self.embedding_module_type = embedding_module_type
    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=None,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory, 
                                                 # don't receive this parameter
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors,
                                                 reuse=self.args.reuse,
                                                 history_budget=self.args.budget,
                                                 args=args,
                                                 num_nodes= self.n_nodes)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,self.n_node_features,1)

  def reset_timer(self):
    self.t_get_message=0
    self.t_store_message=0
    self.t_clear_message=0
    self.t_embedding=0
    self.t_get_memory=0
    self.t_update_memory=0
    self.embedding_module.t_time_encoding=0
    self.embedding_module.t_neighbor=0
    self.embedding_module.t_extra_neighbor=0
    self.embedding_module.t_aggregate=0
    self.embedding_module.t_push_and_pull=0
    self.memory_updater.t_index=0
    self.memory_updater.t_real_update=0
    self.memory_updater.t_others=0
    #self.embedding_module.neighbor_finder.t_find_before=0
    #self.embedding_module.neighbor_finder.t_sample=0


  def get_log(self):
    return self.embedding_module.log

  def print_log(self):
    print(self.embedding_module.log)


  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,edge_idxs, n_neighbors, reuse,cache_plan=None,marker=False):


    self.batch_counter+=1
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    unique_positives = np.unique(positives)
    timestamps = np.concatenate([edge_times, edge_times, edge_times])
    memory = None



    ########## get updated memory ##########
    if self.args.optimize_memory and self.args.new:
      memory = self.memory
    else:
      t_get_memory_start=time.time()
      memory, _ = self.get_updated_memory(self.memory)
      self.t_get_memory+=time.time()-t_get_memory_start

    ############ reuse in the training mode ############ 
    t_embedding_start=time.time()
    if reuse:
      if self.args.new:              
        if self.args.budget!=0:
          node_embedding = self.embedding_module.new_compute_embedding_budget_reuse(memory=memory,source_nodes=nodes,timestamps=timestamps,n_layers=self.n_layers,n_neighbors=n_neighbors,batch_id=self.batch_counter,cache_plan=cache_plan, memory_updater = self.memory_updater)
        else:
          node_embedding = self.embedding_module.new_compute_embedding_reuse(memory=memory,source_nodes=nodes,timestamps=timestamps,n_layers=self.n_layers,n_neighbors=n_neighbors,batch_id=self.batch_counter,memory_updater = self.memory_updater)
      else:
        node_embedding = self.embedding_module.compute_embedding_reuse(memory=memory,source_nodes=nodes,timestamps=timestamps,n_layers=self.n_layers,n_neighbors=n_neighbors,batch_id=self.batch_counter,memory_updater = self.memory_updater)
    else: 
      if self.args.new: 
        node_embedding = self.embedding_module.new_compute_embedding(memory=memory,source_nodes=nodes,timestamps=timestamps,n_layers=self.n_layers,n_neighbors=n_neighbors,memory_updater = self.memory_updater)
      else:
        node_embedding = self.embedding_module.compute_embedding(memory=memory,source_nodes=nodes,timestamps=timestamps,n_layers=self.n_layers,n_neighbors=n_neighbors)
    self.t_embedding+=time.time()-t_embedding_start


    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]


    ########## self.memory is updated ##########
    t_update_memory_start=time.time()

    self.update_memory(self.memory,unique_positives)
    self.t_update_memory+=time.time()-t_update_memory_start


    ######### clear last batch messages #########
    t_clear_message_start=time.time()
    self.memory.clear_messages(unique_positives)
    self.t_clear_message+=time.time()-t_clear_message_start

    ############ message aggregation ############
    t_get_message_start=time.time()
    source_nodes=np.concatenate([source_nodes,destination_nodes])
    destination_nodes=np.concatenate([destination_nodes,source_nodes])
    edge_times=np.concatenate([edge_times,edge_times])
    edge_idxs=np.concatenate([edge_idxs,edge_idxs])
    concat_source_node_embedding=torch.cat((source_node_embedding,destination_node_embedding))
    concat_destination_node_embedding=torch.cat((destination_node_embedding,source_node_embedding))
    unique_sources, source_messages, source_edge_times = self.get_raw_messages(source_nodes,concat_source_node_embedding,destination_nodes,concat_destination_node_embedding,edge_times, edge_idxs)
    self.t_get_message+=time.time()-t_get_message_start

    ############ store raw messages in memory ############
    t_store_message_start=time.time()
    self.memory.store_raw_messages(unique_sources, source_messages, source_edge_times)
    self.t_store_message+=time.time()-t_store_message_start
    return source_node_embedding, destination_node_embedding, negative_node_embedding



  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,edge_idxs, n_neighbors,reuse,cache_plan=None,marker=False):
    #### compute temporal embedding ####
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors,reuse=reuse,cache_plan=cache_plan,marker=marker)

    #### calculate prediction score ####
    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),torch.cat([destination_node_embedding,negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]
    return pos_score.sigmoid(), neg_score.sigmoid()


  ####### update memory #######
  def update_memory(self, memory, positives):
    with torch.no_grad():
      self.memory_updater.update_memory(memory,positives)

  ####### return a copy of updated memory #######
  def get_updated_memory(self, memory):
    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(memory)
    return updated_memory, updated_last_update


  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,destination_node_embedding, edge_times, edge_idxs):

    reversed_source_nodes=np.flip(source_nodes)
    unique_source_nodes,reversed_index=np.unique(reversed_source_nodes,return_index=True)
    index=len(source_nodes)-reversed_index-1
    unique_destination_nodes=destination_nodes[index]
    edge_times=edge_times[index]
    edge_idxs=edge_idxs[index]

    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    # decide to use memory or temporal node embedding, default is memory
    source_memory = self.memory.get_memory(unique_source_nodes) if not self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(unique_destination_nodes) if not self.use_destination_embedding_in_message else destination_node_embedding

    # time delta embedding
    source_time_delta = edge_times - self.memory.last_update[unique_source_nodes,]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(unique_source_nodes,), -1)

    # get message information
    source_message = torch.cat([source_memory, destination_memory, edge_features,source_time_delta_encoding],dim=1)
    return unique_source_nodes, source_message, edge_times


  ######### set neighbor finder as a class member #########
  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
