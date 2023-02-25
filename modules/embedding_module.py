import torch
from torch import nn
import numpy as np
import math
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

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=10, time_diffs=None,use_time_proj=True):
    pass


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=10, time_diffs=None,use_time_proj=True):
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

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=10, time_diffs=None,use_time_proj=True):
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


  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors, memory_updater, train):
  
    assert (n_layers >= 0)
    if n_layers == 0:
      if train:
        index = numba_unique(source_nodes)
        memory,_=memory_updater.get_updated_memory(memory,index)
      source_node_features = memory[source_nodes, :]
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

      combined_embeddings = self.compute_embedding(memory, combined_nodes, combined_timestamps, n_layers-1, n_neighbors, memory_updater,train)
      source_embeddings = combined_embeddings[:n_source_nodes,:]
      neighbor_embeddings = combined_embeddings[n_source_nodes:,:]
      neighbor_embeddings = neighbor_embeddings.view(n_source_nodes, n_neighbors, -1)
      source_embeddings = self.aggregate(n_layers, source_embeddings,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
      return source_embeddings






  def compute_embedding_reuse(self, memory, source_nodes, timestamps, n_layers, n_neighbors, batch_id, memory_updater, train):

    source_nodes_time_embedding = self.time_encoder(torch.zeros((len(timestamps),1),device=self.device))
    neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)

    if train:
      index = numba_unique(np.concatenate((neighbors,source_nodes[:,np.newaxis]),1))
      memory,_=memory_updater.get_updated_memory(memory,index)
    
    neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
    neighbors = neighbors_torch.flatten()
    edge_idxs = torch.from_numpy(edge_idxs).long()
    edge_deltas=timestamps[:, np.newaxis] - edge_times
    edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
    edge_time_embeddings = self.time_encoder(edge_deltas_torch)
    edge_features = self.edge_features[edge_idxs, :]
    mask = neighbors_torch == 0
    
    neighbor_embeddings = memory[neighbors,:]
    neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), n_neighbors, -1)
    source_node_features = memory[source_nodes, :]
    source_embeddings=source_node_features
    for layer_id in range(1,n_layers):
      source_embeddings = self.aggregate(layer_id, source_embeddings,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask.clone())
      neighbor_embeddings=self.push_and_pull(layer_id,source_embeddings,source_nodes,neighbors,batch_id)      
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), n_neighbors, -1)

    source_embeddings = self.aggregate(n_layers, source_embeddings,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
    return source_embeddings



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
    return neighbor_embeddings
   

  def find_uncached_out_of_batch_neighbors(self,source,neighbors,n_layers):
    if n_layers ==1:
      return []

    unique_ngh=numba_unique(neighbors)
    in_index = np.isin(unique_ngh,source) # in-batch neighbor index
    out_index = ~in_index                 # out-of-batch neighbor index
    unique_out_ngh=unique_ngh[out_index]     
    history=self.histories[0]             
    cache_flag=history.cache_flag
    cache_flag=cache_flag[unique_out_ngh]
    index=np.where(cache_flag==0)[0]
    return unique_out_ngh[index][1:]      # filter out dummy node


  def compute_embedding_budget_reuse(self, memory, source_nodes, timestamps, n_layers, n_neighbors,batch_id,cache_plan,memory_updater,train):
    
    if n_layers ==0:
      if train:
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

      combined_embeddings = self.compute_embedding_budget_reuse(memory, combined_nodes, combined_timestamps, n_layers-1, n_neighbors, batch_id,cache_plan,memory_updater,train)
      source_embeddings = combined_embeddings[:n_source_nodes,:]
      neighbor_embeddings = combined_embeddings[n_source_nodes:,:]
      neighbor_embeddings = neighbor_embeddings.view(n_source_nodes, n_neighbors, -1)
      source_embeddings = self.aggregate(n_layers, source_embeddings,source_nodes_time_embedding, neighbor_embeddings,edge_time_embeddings,edge_features,mask)
      return source_embeddings

    else:
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)

      uncached_neighbors= self.find_uncached_out_of_batch_neighbors(source_nodes,neighbors, n_layers)
      
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
      
      # get embeddings of source nodes and uncached neighbor nodes
      combined_embeddings = self.compute_embedding_budget_reuse(memory, combined_nodes, combined_timestamps, n_layers-1, n_neighbors, batch_id,cache_plan,memory_updater,train)
      
      # extract embeddings of source nodes
      n_source_nodes = len(source_nodes)      
      source_embeddings=combined_embeddings[:n_source_nodes,:]

      # extract embeddings of neighbor nodes
      neighbor_embeddings=self.budget_push_and_pull(n_layers-1,combined_embeddings,combined_nodes,neighbors,batch_id,cache_plan)
      neighbor_embeddings = neighbor_embeddings.view(n_source_nodes, n_neighbors, -1)

      # feature aggregation
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


