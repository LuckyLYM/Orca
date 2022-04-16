from ast import arg
from collections import UserString
from zmq import device
from numpy.core.fromnumeric import argmax
from numpy.lib.arraysetops import unique
from numpy.lib.utils import source
import torch
from torch import nn
import numpy as np
import math
import time
import sys
from modules.history import History
from model.temporal_attention import TemporalAttentionLayer
from numba import njit


@njit
def numba_unique(nodes):
  return np.unique(nodes)

class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device
    self.log=dict()

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,use_time_proj=True):
    pass


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))
    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, reuse=False, history_budget=0,args=None, num_nodes = -1):
    
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)
    self.use_memory = use_memory
    self.device = device
    self.args=args
    self.budget=history_budget


    self.t_time_encoding=0
    self.t_neighbor=0
    self.t_aggregate=0
    self.t_push_and_pull=0
    self.t_extra_neighbor=0
    #num_nodes=node_features.shape[0]

    if reuse:
      self.histories = torch.nn.ModuleList(
      [History(num_nodes, embedding_dimension, device,history_budget)
      for _ in range(n_layers - 1)])

  def backup_history(self):
    n_layers=self.n_layers
    backup=[self.histories[i].emb.clone() for i in range(n_layers - 1)]
    return backup

  def restore_history(self,backup):
    n_layers=self.n_layers
    for i in range(n_layers-1):
      self.histories[i].emb=backup[i]

  def reset_history(self):
    n_layers=self.n_layers
    for i in range(n_layers-1):
      self.histories[i].reset_parameters()

  def detach_history(self):
    n_layers=self.n_layers
    for i in range(n_layers-1):
      self.histories[i].detach_history()



  def push_and_pull(self,layer_id,source_embedding,source_nodes,neighbors,batch_id):
    embeddings=source_embedding
    ids=source_nodes
    history=self.histories[layer_id-1]    
    history.update_times[ids]=batch_id
    if not self.args.gradient:
      history.push(embeddings,ids)
      neighbor_embeddings=history.pull(neighbors)
    else:
      neighbor_embeddings=history.push_and_pull(embeddings,ids,neighbors)
    return neighbor_embeddings



  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors):
    assert (n_layers >= 0)
    source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))
    source_node_features = memory[source_nodes, :]

    if n_layers == 0:
      return source_node_features

    neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)
    neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
    edge_deltas = timestamps[:, np.newaxis] - edge_times
    edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
    neighbors = neighbors.flatten() 

    neighbor_embeddings = self.compute_embedding(memory,neighbors,np.repeat(timestamps, n_neighbors),n_layers=n_layers - 1,n_neighbors=n_neighbors)
    effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1

    neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
    edge_time_embeddings = self.time_encoder(edge_deltas_torch)
    edge_features = self.edge_features[edge_idxs, :]
    mask = neighbors_torch == 0
    source_embedding = self.aggregate(n_layers, source_node_features,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
    return source_embedding


  def compute_embedding_reuse(self, memory, source_nodes, timestamps, n_layers, n_neighbors,batch_id):

    source_node_features = memory[source_nodes, :]
    source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))

    neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)
    neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    neighbors = neighbors_torch.flatten() # row by row flatten
    neighbor_embeddings=memory[neighbors,:]
    neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), n_neighbors, -1)
    edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
    edge_deltas = timestamps[:, np.newaxis] - edge_times
    edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
    edge_time_embeddings = self.time_encoder(edge_deltas_torch)
    edge_features = self.edge_features[edge_idxs, :]
    mask = neighbors_torch == 0

    for layer_id in range(1,n_layers):
      source_embedding = self.aggregate(layer_id, source_node_features,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask.clone())
      neighbor_embeddings=self.push_and_pull(layer_id,source_embedding,source_nodes,neighbors,batch_id)
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), n_neighbors, -1)

    source_embedding = self.aggregate(n_layers, source_node_features,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
    return source_embedding
  


  #################### new version: using temporal embeddings as queries ####################
  def new_compute_embedding_backup(self, memory, source_nodes, timestamps, n_layers, n_neighbors, memory_updater):

    assert (n_layers >= 0)
    if n_layers == 0:
      #source_node_features = memory[source_nodes, :] + self.node_features[source_nodes, :]
      #sys.exit(1)
      source_node_features = memory[source_nodes, :]
      return source_node_features
    else:
      assert(n_neighbors==10)
      source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)
      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
      edge_deltas = timestamps[:, np.newaxis] - edge_times
      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten() 
      source_node_features=self.new_compute_embedding(memory, source_nodes, timestamps, n_layers-1, n_neighbors,memory_updater)
      neighbor_embeddings = self.new_compute_embedding(memory,neighbors,np.repeat(timestamps, n_neighbors),n_layers - 1, n_neighbors,memory_updater)
      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1

      # get neighbor embeddings
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)
      edge_features = self.edge_features[edge_idxs, :]
      mask = neighbors_torch == 0
      source_embedding = self.aggregate(n_layers, source_node_features,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
      return source_embedding


  def new_compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors, memory_updater):
  
    assert (n_layers >= 0)
    if n_layers == 0:
      if self.args.optimize_memory:
        #index = np.unique(source_nodes)
        index = numba_unique(source_nodes)
        memory,_=memory_updater.get_updated_memory(memory,index)
      source_node_features = memory[source_nodes, :]
      #source_node_features = memory[source_nodes, :] + self.node_features[source_nodes, :]
      return source_node_features
    else:
      source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)
      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
      edge_deltas = timestamps[:, np.newaxis] - edge_times
      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)
      edge_features = self.edge_features[edge_idxs, :]
      mask = neighbors_torch == 0

      neighbors = neighbors.flatten() 
      combined_nodes = np.hstack((source_nodes,neighbors))
      neighbor_timestamps = np.repeat(timestamps, n_neighbors)
      combined_timestamps = np.hstack((timestamps,neighbor_timestamps))
      n_source_nodes = len(source_nodes)

      combined_embeddings = self.new_compute_embedding(memory, combined_nodes, combined_timestamps, n_layers-1, n_neighbors, memory_updater)

      source_embeddings = combined_embeddings[:n_source_nodes,:]
      neighbor_embeddings = combined_embeddings[n_source_nodes:,:]
      neighbor_embeddings = neighbor_embeddings.view(n_source_nodes, n_neighbors, -1)
      source_embeddings = self.aggregate(n_layers, source_embeddings,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
      return source_embeddings





  def new_compute_embedding_reuse(self, memory, source_nodes, timestamps, n_layers, n_neighbors, batch_id , memory_updater):

    source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))
    neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)

    if self.args.optimize_memory:
      #index = np.unique(np.concatenate((neighbors,source_nodes[:,np.newaxis]),1))
      index = numba_unique(np.concatenate((neighbors,source_nodes[:,np.newaxis]),1))

      memory,_=memory_updater.get_updated_memory(memory,index)
    source_node_features = memory[source_nodes, :]

    neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    neighbors = neighbors_torch.flatten()
    neighbor_embeddings = memory[neighbors,:]
    neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), n_neighbors, -1)

    edge_idxs = torch.from_numpy(edge_idxs).long()
    edge_deltas=timestamps[:, np.newaxis] - edge_times
    edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
    edge_time_embeddings = self.time_encoder(edge_deltas_torch)

    edge_features = self.edge_features[edge_idxs, :]
    mask = neighbors_torch == 0
    source_embeddings=source_node_features


    for layer_id in range(1,n_layers):
      source_embeddings = self.aggregate(layer_id, source_embeddings,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask.clone())

      neighbor_embeddings=self.push_and_pull(layer_id,source_embeddings,source_nodes,neighbors,batch_id)      
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), n_neighbors, -1)

    source_embeddings = self.aggregate(n_layers, source_embeddings,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
    return source_embeddings




  # TODO: check this, anything wrong with this function???
  def budget_push_and_pull(self,layer_id,source_embedding,source_nodes,neighbors,batch_id,cache_plan):
    embeddings=source_embedding
    ids=source_nodes
    history=self.histories[layer_id-1]
    history.update_times[ids]=batch_id

    if not self.args.gradient:
      history.push(embeddings,ids)
      neighbor_embeddings=history.pull(neighbors)
    else:
      neighbor_embeddings=history.push_and_pull(embeddings,ids,neighbors)
    history.update_flag(cache_plan,source_nodes)

    
    debug = False
    if debug:
      cached=history.cache_flag
      cached=np.where(cached==1)[0]
      candidate=np.concatenate((source_nodes,cached))
      if cache_plan is not None:
        for node in cache_plan:
          assert(node in candidate)

    return neighbor_embeddings
   

  # ! In fact, no need to search. Just caching and reusing the cache plan results is fine.
  def find_uncached_out_of_batch_neighbors(self,source,neighbors,n_layers):
    if n_layers ==1:
      return []
    unique_ngh=numba_unique(neighbors)
    
    in_index = np.isin(unique_ngh,source) # true and false
    out_index = ~in_index
    unique_ngh=unique_ngh[out_index]

    history=self.histories[0]
    cache_flag=history.cache_flag
    cache_flag=cache_flag[unique_ngh]
    index=np.where(cache_flag==0)[0]
    return unique_ngh[index][1:]


  def new_compute_embedding_budget_reuse(self, memory, source_nodes, timestamps, n_layers, n_neighbors,batch_id,cache_plan,memory_updater):
    assert (n_layers >= 0)
    if n_layers ==0:
      if self.args.optimize_memory:
        index = numba_unique(source_nodes)
        memory,_=memory_updater.get_updated_memory(memory,index)
      source_node_features = memory[source_nodes, :]
      return source_node_features

    elif n_layers ==1:
      source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)
      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
      edge_deltas = timestamps[:, np.newaxis] - edge_times
      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)
      edge_features = self.edge_features[edge_idxs, :]
      mask = neighbors_torch == 0

      neighbors = neighbors.flatten() 
      combined_nodes = np.hstack((source_nodes,neighbors))
      neighbor_timestamps = np.repeat(timestamps, n_neighbors)
      combined_timestamps = np.hstack((timestamps,neighbor_timestamps))
      n_source_nodes = len(source_nodes)

      combined_embeddings = self.new_compute_embedding_budget_reuse(memory, combined_nodes, combined_timestamps, n_layers-1, n_neighbors, batch_id,cache_plan,memory_updater)

      source_embeddings = combined_embeddings[:n_source_nodes,:]
      neighbor_embeddings = combined_embeddings[n_source_nodes:,:]
      neighbor_embeddings = neighbor_embeddings.view(n_source_nodes, n_neighbors, -1)
      source_embeddings = self.aggregate(n_layers, source_embeddings,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
      return source_embeddings

    else:
      # ! the source nodes here include fake target nodes
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)
      uncached_neighbors= self.find_uncached_out_of_batch_neighbors(source_nodes,neighbors, n_layers)

      #print(f'uncached {len(uncached_neighbors)}')
      if len(uncached_neighbors)!=0:
        max_timestamp=np.max(timestamps)
        extra_timestamps=np.repeat(max_timestamp,len(uncached_neighbors))
        combined_nodes=np.hstack((source_nodes,uncached_neighbors))
        combined_timestamps=np.hstack((timestamps,extra_timestamps))
      else:
        combined_nodes=source_nodes
        combined_timestamps=timestamps

      source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))    
      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
      neighbors = neighbors_torch.flatten()
      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
      edge_deltas = timestamps[:, np.newaxis] - edge_times
      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)
      edge_features = self.edge_features[edge_idxs, :]
      mask = neighbors_torch == 0
      
      # get source
      n_source_nodes = len(source_nodes)
      combined_embeddings = self.new_compute_embedding_budget_reuse(memory, combined_nodes, combined_timestamps, n_layers-1, n_neighbors, batch_id,cache_plan,memory_updater)
      source_embeddings=combined_embeddings[:n_source_nodes,:]

      # pull neighbor
      neighbor_embeddings=self.budget_push_and_pull(n_layers-1,combined_embeddings,combined_nodes,neighbors,batch_id,cache_plan)
      neighbor_embeddings = neighbor_embeddings.view(n_source_nodes, n_neighbors, -1)

      source_embeddings = self.aggregate(n_layers, source_embeddings, source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
      return source_embeddings




  def new_compute_embedding_budget_reuse_backup(self, memory, source_nodes, timestamps, n_layers, n_neighbors,batch_id, cache_plan,memory_updater):

    neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)
    extra_neighbors= self.find_uncached_neighbors(neighbors,n_layers)
    
    if len(extra_neighbors)!=0:
      max_timestamp=np.max(timestamps)
      extra_timestamps=np.repeat(max_timestamp,len(extra_neighbors))
      neighbors2, edge_idxs2, edge_times2 = self.neighbor_finder.get_temporal_neighbor(extra_neighbors,extra_timestamps,n_neighbors)

      combined_source_nodes=np.hstack((source_nodes,extra_neighbors))
      combined_timestamps=np.hstack((timestamps,extra_timestamps))
      combined_neighbors=np.vstack((neighbors,neighbors2))
      combined_edge_idxs=np.vstack((edge_idxs,edge_idxs2))
      combined_edge_times=np.vstack((edge_times,edge_times2))
    else:
      combined_source_nodes=source_nodes
      combined_timestamps=timestamps
      combined_neighbors=neighbors
      combined_edge_idxs=edge_idxs
      combined_edge_times=edge_times


    if self.args.optimize_memory:
      #index = np.unique(np.concatenate((combined_neighbors,combined_source_nodes[:,np.newaxis]),1))
      index = numba_unique(np.concatenate((combined_neighbors,combined_source_nodes[:,np.newaxis]),1))
      memory,_=memory_updater.get_updated_memory(memory,index)

    #neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    #neighbors = neighbors_torch.flatten() 


    #combined_source_node_features = memory[combined_nodes, :] + self.node_features[combined_nodes]
    combined_source_node_features = memory[combined_source_nodes, :]
    combined_source_nodes_time_embedding = self.time_encoder(torch.zeros((len(combined_timestamps),1),device=self.device))    
    

    combined_neighbors_torch = torch.from_numpy(combined_neighbors).long().to(self.device)
    combined_neighbors = combined_neighbors_torch.flatten()
    #combined_neighbor_embeddings = memory[combined_neighbors, :] + self.node_features[combined_neighbors]
    combined_neighbor_embeddings = memory[combined_neighbors,:]
    combined_neighbor_embeddings = combined_neighbor_embeddings.view(len(combined_source_nodes), n_neighbors, -1)
    
    # shape: [660,1] - [660,10]
    #print(f'combined_timestamps {combined_timestamps.shape},  combined_edge_times {combined_edge_times.shape}')
    combined_edge_deltas = combined_timestamps[:, np.newaxis] - combined_edge_times
    combined_edge_deltas_torch = torch.from_numpy(combined_edge_deltas).float().to(self.device)
    combined_edge_time_embeddings = self.time_encoder(combined_edge_deltas_torch)
    #print(f'combined_edge_time_embeddings {combined_edge_time_embeddings.shape}')
    # [660,10,172]

    # neighboring edge features of combined nodes
    combined_edge_idxs = torch.from_numpy(combined_edge_idxs).long()
    combined_edge_features = self.edge_features[combined_edge_idxs, :]
    combined_mask = combined_neighbors_torch == 0

    # get embeddings for 1-layer target nodes
    n_samples=len(source_nodes)//3
    source_nodes_time_embedding=combined_source_nodes_time_embedding[:3*n_samples]
    edge_time_embeddings=combined_edge_time_embeddings[:3*n_samples,:]
    edge_features=combined_edge_features[:3*n_samples,:]
    neighbors = combined_neighbors[:3*n_samples*n_neighbors].clone()
    mask=combined_mask[:3*n_samples].clone()
    #print(f'edge_time_embeddings {edge_time_embeddings.shape}')
    # [660,10,172]


    # ! we currently only support 2-layer models
    combined_source_embeddings=combined_source_node_features
    for layer_id in range(1,n_layers):
      combined_source_embeddings = self.aggregate(layer_id, combined_source_embeddings, combined_source_nodes_time_embedding, combined_neighbor_embeddings,combined_edge_time_embeddings,combined_edge_features,combined_mask)
      neighbor_embeddings=self.budget_push_and_pull(layer_id,combined_source_embeddings,combined_source_nodes,neighbors,batch_id,cache_plan)
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), n_neighbors, -1)
      source_embeddings=combined_source_embeddings[:n_samples*3]

    source_embeddings = self.aggregate(n_layers, source_embeddings, source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
    return source_embeddings





  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,neighbor_embeddings,edge_time_embeddings, edge_features, mask):
    return None



class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,n_node_features, n_edge_features, n_time_features, embedding_dimension, device,n_heads=2, dropout=0.1, use_memory=True,reuse=False,history_budget=-1,args=None, num_nodes = -1):

    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory,reuse,history_budget,args,num_nodes)

    self.attention_models = torch.nn.ModuleList(
      [TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)]
      )


  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings, edge_time_embeddings, edge_features, mask):

    attention_model = self.attention_models[n_layer - 1]

    # shape: [batch_size, n_neighbors]
    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)
    return source_embedding






class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +n_edge_features, embedding_dimension) for _ in range(n_layers)])
    
    self.linear_2 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,neighbor_embeddings,edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))
    source_features = torch.cat([source_node_features,source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True,reuse=False,history_budget=0,args=None, num_nodes = -1):

  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                    reuse=reuse,
                                    history_budget=history_budget,args=args,num_nodes=num_nodes)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


