import numpy as np
from numpy.lib.histograms import _search_sorted_inclusive
import torch
import math 
import random
import time
from numba import jit
from numba.experimental import jitclass
from numba import int32, float32, types, typed
import numba as nb
from tqdm import tqdm

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0
    self.epoch_count = 0
    self.best_epoch = 0
    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1
    self.epoch_count += 1
    return self.num_round >= self.max_round

class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)
    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)
  def sample(self, size):
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), size)
      dst_index = np.random.randint(0, len(self.dst_list), size)
    else:
      src_index = self.random_state.randint(0, len(self.src_list), size)
      dst_index = self.random_state.randint(0, len(self.dst_list), size)
    return self.src_list[src_index], self.dst_list[dst_index]
  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)



def get_neighbor_finder(data, uniform, max_node_idx=None,time_encoding ='real',unique=False):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,data.edge_idxs, data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))
  node_to_neighbors =typed.List()
  node_to_edge_idxs = typed.List()
  node_to_edge_timestamps = typed.List()
  for neighbors in adj_list:
    sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
    node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors],dtype=np.int32))
    node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors],dtype=np.int32))
    node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors],dtype=np.float64))
  return NeighborFinder(node_to_neighbors,node_to_edge_idxs,node_to_edge_timestamps)



l_int = typed.List()
l_float = typed.List()
a_int=np.array([1,2],dtype=np.int32)
a_float=np.array([1,2],dtype=np.float64)
l_int.append(a_int)
l_float.append(a_float)
spec = [
    ('node_to_neighbors', nb.typeof(l_int)),        
    ('node_to_edge_idxs', nb.typeof(l_int)),      
    ('node_to_edge_timestamps', nb.typeof(l_float)),       
]
@jitclass(spec)
class NeighborFinder:
  def __init__(self,node_to_neighbors,node_to_edge_idxs,node_to_edge_timestamps):
    self.node_to_neighbors = node_to_neighbors
    self.node_to_edge_idxs = node_to_edge_idxs
    self.node_to_edge_timestamps = node_to_edge_timestamps
    #self.t_find_before=0
    #self.t_sample=0

  def find_before(self, src_idx, cut_time):
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]


  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors):

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors),dtype=np.int32)  
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors),dtype=np.float32) 
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors),dtype=np.int32)

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      #t_find_before_start=time.time()
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,timestamp)  
      #self.t_find_before+=time.time()-t_find_before_start

      #t_sample_start=time.time()
      if len(source_neighbors) > 0 and n_neighbors > 0:
        source_edge_times = source_edge_times[-n_neighbors:]
        source_neighbors = source_neighbors[-n_neighbors:]
        source_edge_idxs = source_edge_idxs[-n_neighbors:]
        n_ngh=len(source_neighbors)
        neighbors[i, n_neighbors - n_ngh:] = source_neighbors
        edge_idxs[i, n_neighbors - n_ngh:] = source_edge_idxs
        edge_times[i, n_neighbors - n_ngh:] = source_edge_times

      #self.t_sample+=time.time()-t_sample_start
    return neighbors, edge_idxs, edge_times







###################### following functions are used to get cache plan ######################
###################### following functions are used to get cache plan ######################
def read_batches(args,full_train_data,neighbor_finder,num_embeddings):
  BATCH_SIZE = args.bs
  strategy=args.sampling_strategy
  ratio=args.ratio
  n_degree=args.n_degree
  budget=args.budget

  if strategy=='epoch':
    train_data=full_train_data.sample(ratio)
  else:
    train_data = full_train_data

  num_instance = len(train_data.sources)
  if strategy=='partition':
    num_batch = train_data.n_batch
  else:
    num_batch = math.ceil(num_instance/BATCH_SIZE)

  target_list=[]
  ngh_list=[]

  total_n_in=0
  total_n_unique_in=0
  total_n_out=0
  total_n_unique_out=0

  target_batches=dict()
  ngh_batches=dict()

  for i in range(num_embeddings):
    target_batches[i]=[]
    ngh_batches[i]=[]

  for batch_idx in tqdm(range(0, num_batch)):
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

    sources_batch, destinations_batch = train_data.sources[sample_inds],train_data.destinations[sample_inds]
    timestamps_batch = train_data.timestamps[sample_inds]

    # ! Yiming: we don't consider negative sampled nodes here...
    source_nodes = np.concatenate([sources_batch, destinations_batch])
    timestamps = np.concatenate([timestamps_batch, timestamps_batch])
    neighbors, _, _ = neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_degree)
    neighbors=neighbors[neighbors!=0]  #[400,10] => 1 dimensional array

    n_target = len(source_nodes)
    n_neighbors=len(neighbors)
    unique_target=np.unique(source_nodes)
    unique_neighbors=np.unique(neighbors)


    unique_in = np.intersect1d(unique_target, unique_neighbors)
    in_index = np.isin(neighbors,unique_in) # true and false
    n_in = np.count_nonzero(in_index)
    n_unique_in = len(unique_in)
    total_n_in += n_in
    total_n_unique_in+=n_unique_in


    # get out
    out_index = ~in_index
    out = neighbors[out_index]
    unique_out=np.unique(out)

    n_out= len(out)
    n_unique_out = len(unique_out)  
    total_n_out += n_out
    total_n_unique_out+=n_unique_out

    #print(f'target {n_target}, neighbor {n_neighbors}, n_in {n_in}, n_unique_in {n_unique_in}, n_out {n_out}, n_unique_out {n_unique_out}')
    assert((n_in+n_out)==n_neighbors)

    target_list.append(unique_target)

    # TODO: it is not unique here
    ngh_list.append(out)

    # * batch id where nodes appear as target nodes
    for target in unique_target:
      target_batches[target].append(batch_idx)

    # * batch id where nodes appear as 1-hop out-of-batch nodes and can be reused
    for ngh in unique_out:
      ngh_batches[ngh].append(batch_idx)

  # target_list: target nodes in each batch
  # ngh_list: out of batch nodes in each batch
  # target_batches: appear batch ids of each target node
  # ngh_batches: appear batch ids of each neighbor node
  print(f'n_in {total_n_in}, n_unique_in {total_n_unique_in}, n_out {total_n_out}, n_unique_out {total_n_unique_out}')
  return num_batch,target_list,ngh_list,target_batches,ngh_batches,total_n_in,total_n_unique_in,total_n_out,total_n_unique_out




# TODO: go through my code and check whether any bug exists, I think so
def get_cache_plan_FIF(args,cache_flag,full_train_data,neighbor_finder,num_embeddings):
  # target_list: target nodes in each batch
  # ngh_list: out of batch nodes in each batch
  # target_batches: appear batch ids of each target node
  # ngh_batches: appear batch ids of each neighbor node
  budget=args.budget
  num_batch,target_list,ngh_list,target_batches,ngh_batches,total_n_in,total_n_unique_in,total_n_out,total_n_unique_out=read_batches(args,full_train_data,neighbor_finder,num_embeddings)

  n_reuse=0
  n_recompute=0
  n_out=0
  total_reuse_distance = 0
  time_flag=np.zeros(num_embeddings)
  MAX_DISTANCE=100000000

  cache_plan_list=[]

  for batch_idx in tqdm(range(num_batch)):
    target=target_list[batch_idx]
    ngh=ngh_list[batch_idx]   # ! Yiming: this is not unique here
    ngh_size=len(ngh)
    n_out+=ngh_size
    
    cache_=cache_flag[ngh]
    index=np.where(cache_==0)[0]
    uncached_ngh=ngh[index] 

    # TODO: we can actually use the uncached_ngh here and don't search it again in the training stage


    # TODO: why there will be n_recompute, how about the nodes appear in the first time
    # ! I see there are out-of-batch neighbor nodes
    n_recompute+=len(uncached_ngh)

    index=np.where(cache_==1)[0]
    cached_ngh=ngh[index]
    batch_reuse_distance = np.sum(batch_idx - time_flag[cached_ngh])
    #print(f'sum {batch_reuse_distance}, n {len(cached_ngh)} {batch_reuse_distance/len(cached_ngh)}')
    
    total_reuse_distance+= batch_reuse_distance
    n_reuse+=len(cached_ngh)
    assert(ngh_size==len(uncached_ngh)+len(cached_ngh))


    # * union of already cached, uncached neighbors, newly computed
    cached=np.where(cache_flag==1)[0]
    new_computed = np.concatenate((uncached_ngh,target))
    new_computed = np.unique(new_computed)
    candidates=np.concatenate((uncached_ngh,cached,target))
    candidates=np.unique(candidates)


    # * analyze the next reuse time of each candidate node
    reuse_distance_list=[]
    for index, node in enumerate(candidates):      
      # filter out fake nodes such as zero paddings
      target_ids=np.array(target_batches[node])
      if len(target_ids)==0:
        assert(node==0)
        reuse_distance_list.append(MAX_DISTANCE+1)
        continue
      

      # endtime analysis
      # log(E)
      end_index=np.argmax(target_ids>batch_idx)
      end_time=target_ids[end_index]
      if end_time<=batch_idx: # extend endtime to the end of the batch
        end_time=num_batch+1
      end_time=end_time-1

      # reuse time analysis
      # log(E)
      reuse_ids=np.array(ngh_batches[node])
      reuse_index=np.where(np.logical_and(reuse_ids>batch_idx, reuse_ids<=end_time))[0]
      reuse_times=len(reuse_index)
      if reuse_times==0:
        reuse_distance_list.append(MAX_DISTANCE)
      else:
        reuse_distance_list.append(reuse_ids[reuse_index][0])

    # * sort by reuse distance
    reuse_distance_list=np.array(reuse_distance_list)
    sorted_inds=np.argsort(reuse_distance_list) # sort in increasing order
    sorted_reuse_distance=reuse_distance_list[sorted_inds]
    sorted_nodes=candidates[sorted_inds]

    # * the cache plan means what to materialize after processing a new batch
    if len(sorted_nodes)!=0:
      to_cache=sorted_nodes[:budget]  # to be cached node ids
      cache_flag=np.zeros(num_embeddings)
      cache_flag[to_cache]=1
      cache_plan_list.append(to_cache)

      new_index = np.isin(to_cache,new_computed)
      new_nodes = to_cache[new_index]
      time_flag[new_nodes]=batch_idx
    else:
      cache_plan_list.append(None)
  
  # ! seems so far so good

  assert(n_out==total_n_out)
  print(f'method FIF, budget {budget}, n_out_reuse {n_reuse}, n_out_recompute {n_recompute}, n_in_reuse {total_n_in}, reuse_distance {total_reuse_distance/n_reuse}')
  return cache_plan_list, n_reuse, n_recompute, total_n_in,total_reuse_distance














































def read_batches_multiple_stages(args,data_list,neighbor_finder,num_embeddings):
  BATCH_SIZE = args.bs
  strategy=args.sampling_strategy
  ratio=args.ratio
  n_neighbors=args.n_degree
  num_batch_list=[]

  target_list=[]
  ngh_list=[]
  target_batches=dict()
  ngh_batches=dict()
  for i in range(num_embeddings):
    target_batches[i]=[]
    ngh_batches[i]=[]
  previous_batch=0
  batch_end=0

  for data in data_list:

    if strategy=='epoch':
      train_data=data.sample(ratio)
    else:
      train_data = data
    num_instance = len(train_data.sources)
    if strategy=='partition':
      num_batch = train_data.n_batch
    else:
      num_batch = math.ceil(num_instance/BATCH_SIZE)
    num_batch_list.append(num_batch)
    batch_start=batch_end
    batch_end=batch_start+num_batch
    print(f'batch_start {batch_start}, batch_end {batch_end}')


    for batch_idx in range(0, num_batch):

      # get index of a training batch
      start_idx = batch_idx * BATCH_SIZE
      end_idx = min(num_instance, start_idx + BATCH_SIZE)
      sample_inds=np.array(list(range(start_idx,end_idx)))

      if strategy=='batch':
        batch_size=end_idx-start_idx
        sample_size=int(batch_size*ratio)
        sample_inds=random.sample(range(start_idx,end_idx),sample_size)

      elif strategy=='partition':
        batch_inds=train_data.tbatch[batch_idx]
        sample_size=int(len(batch_inds)*ratio)
        sample_inds=random.sample(batch_inds,sample_size)

      # * load a batch of training data, we don't consider negative sampling here
      sources_batch, destinations_batch = train_data.sources[sample_inds],train_data.destinations[sample_inds]
      edge_idxs_batch = train_data.edge_idxs[sample_inds]
      timestamps_batch = train_data.timestamps[sample_inds]
      size = len(sources_batch)

      source_nodes = np.concatenate([sources_batch, destinations_batch])
      timestamps = np.concatenate([timestamps_batch, timestamps_batch])
      neighbors, _, _ = neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_neighbors)

      # analyze the batch distribution
      unique_target=np.unique(source_nodes)
      unique_ngh=np.unique(neighbors)
      unique_ob_ngh=np.setdiff1d(unique_ngh,unique_target,True)

      target_list.append(unique_target)
      ngh_list.append(unique_ob_ngh)

      # * batch id where nodes appear as target nodes
      for target in unique_target:
        target_batch_id=batch_idx+batch_start
        assert(target_batch_id>=batch_start and target_batch_id<batch_end)
        target_batches[target].append(target_batch_id)

      # * batch id where nodes appear as 1-hop out-of-batch nodes and can be reused
      for ngh in unique_ob_ngh:
        target_batch_id=batch_idx+batch_start
        assert(target_batch_id>=batch_start and target_batch_id<batch_end)
        ngh_batches[ngh].append(target_batch_id)


  print(f'read batch multiple stages {num_batch_list}')
  return num_batch_list,target_list,ngh_list,target_batches,ngh_batches




def get_cache_plan_FIF_multiple_stages(args,cache_flag,data_list,neighbor_finder,num_embeddings):

  budget=args.budget
  num_batch_list,target_list,ngh_list,target_batches,ngh_batches=read_batches_multiple_stages(args,data_list,neighbor_finder,num_embeddings)
  num_batch=np.sum(num_batch_list)

  n_reuse=0
  n_recompute=0
  cache_plan_list=[]
  MAX_DISTANCE=100000000


  for batch_idx in range(num_batch):
    target=target_list[batch_idx]
    ngh=ngh_list[batch_idx]
    ngh_size=len(ngh)
    cache_=cache_flag[ngh]

    index=np.where(cache_==0)[0]
    uncached_ngh=ngh[index]
    n_recompute+=len(uncached_ngh)

    index=np.where(cache_==1)[0]
    cached_ngh=ngh[index]
    n_reuse+=len(cached_ngh)
    assert(ngh_size==len(uncached_ngh)+len(cached_ngh))


    # * union of already cached, uncached neighbors, newly computed
    cached=np.where(cache_flag==1)[0]
    candidates=np.concatenate((uncached_ngh,cached,target))
    candidates=np.unique(candidates)


    # * analyze the next reuse time
    reuse_distance_list=[]
    for index, node in enumerate(candidates):      
      # filter out fake nodes such as zero paddings, this node is never appeared as target nodes
      target_ids=np.array(target_batches[node])
      if len(target_ids)==0:

        # TODO: why there are so many fake nodes? all happens in the test and val stages, 
        # TODO: and they are all the uncached 1-hop out-of-batch neighbors...
        # * The reason is dismatch between training sets and neighbor finder, we have manually remove some nodes for the inductive test
        #assert(node==0)
        '''
        if node!=0:
          #print(f'batch_id {batch_idx}, node {node}')
          #print(f'in target {node in target}, in uncached {node in uncached_ngh}, in cached {node in cached}')
          #assert(node in uncached_ngh)
          #assert(node not in data_list[0].unique_nodes)
          #assert(node not in data_list[1].unique_nodes)
          #assert(node not in data_list[2].unique_nodes)
        '''

        

        reuse_distance_list.append(MAX_DISTANCE+1)
        continue
      

      # endtime analysis
      end_index=np.argmax(target_ids>batch_idx)
      end_time=target_ids[end_index]
      if end_time<=batch_idx: # extend endtime to the end of the batch
        end_time=num_batch+1
      end_time=end_time-1

      # reuse time analysis
      reuse_ids=np.array(ngh_batches[node])
      reuse_index=np.where(np.logical_and(reuse_ids>batch_idx, reuse_ids<=end_time))[0]
      reuse_times=len(reuse_index)
      if reuse_times==0:
        reuse_distance_list.append(MAX_DISTANCE)
      else:
        reuse_distance_list.append(reuse_ids[reuse_index][-1])

    # * sort by reuse distance
    reuse_distance_list=np.array(reuse_distance_list)
    sorted_inds=np.argsort(reuse_distance_list)
    sorted_reuse_distance=reuse_distance_list[sorted_inds]
    sorted_nodes=candidates[sorted_inds]

    # * the cache plan means what to materialize after processing a new batch
    if len(sorted_nodes)!=0:
      to_cache=sorted_nodes[:budget]
      cache_flag=np.zeros(num_embeddings)
      cache_flag[to_cache]=1
      cache_plan_list.append(to_cache)
    else:
      cache_plan_list.append(None)

  print(f'method FIF, n_reuse {n_reuse}, n_recompute {n_recompute}')

  #print(f'num_batch_list {num_batch_list}')
  #print(f'cache_plan_list {len(cache_plan_list)}')
  if len(num_batch_list)==1:
    cache_plans=cache_plan_list
  else:
    start=0
    cache_plans=[]
    for num_batch in num_batch_list:
      end=start+num_batch
      #print(f'start {start}, end {end}')
      cache_plans.append(cache_plan_list[start:end])
      start=end


  return cache_plans, n_reuse, n_recompute








'''
def get_cache_plan_reuse_ratio(args,cache_flag,full_train_data,neighbor_finder,num_embeddings):

  budget=args.budget
  num_batch,target_list,ngh_list,target_batches,ngh_batches=read_batches(args,full_train_data,neighbor_finder,num_embeddings)


  # TODO: we should remove the effect of padding nodes here, let's check it out...

  n_reuse=0
  n_recompute=0
  cache_plan_list=[]

  n_out=0

  for batch_idx in range(num_batch):
    target=target_list[batch_idx]
    ngh=ngh_list[batch_idx]
    ngh_size=len(ngh)
    cache_=cache_flag[ngh]
    n_out+=ngh_size

    index=np.where(cache_==0)[0]
    uncached_ngh=ngh[index]
    n_recompute+=len(uncached_ngh)

    index=np.where(cache_==1)[0]
    cached_ngh=ngh[index]
    n_reuse+=len(cached_ngh)
    assert(ngh_size==len(uncached_ngh)+len(cached_ngh))


    # * union of already cached, uncached neighbors, newly computed
    cached=np.where(cache_flag==1)[0]
    candidates=np.concatenate((uncached_ngh,cached,target))
    candidates=np.unique(candidates)


    # * analyze the next reuse time
    candidate_list=[]
    reuse_ratio_list=[]
    for index, node in enumerate(candidates):      
      # filter out fake nodes such as zero paddings
      target_ids=np.array(target_batches[node])
      if len(target_ids)==0:
        assert(node==0)
        continue
      
      # endtime analysis
      end_index=np.argmax(target_ids>batch_idx)
      end_time=target_ids[end_index]
      if end_time<=batch_idx: # extend endtime to the end of the batch
        end_time=num_batch+1
      end_time=end_time-1


      # reuse time analysis
      reuse_ids=np.array(ngh_batches[node])
      reuse_index=np.where(np.logical_and(reuse_ids>batch_idx, reuse_ids<=end_time))[0]
      reuse_times=len(reuse_index)
      if reuse_times==0:
        continue
      else:
        lifetime=reuse_ids[reuse_index][-1]-batch_idx
        reuse_ratio=reuse_times/lifetime
        candidate_list.append(node)
        reuse_ratio_list.append(reuse_ratio)

    reuse_ratio_list=np.array(reuse_ratio_list)
    candidate_list=np.array(candidate_list)
    sorted_inds=np.argsort(-reuse_ratio_list)
    sorted_reuse_ratio=reuse_ratio_list[sorted_inds]
    sorted_nodes=candidates[sorted_inds]

    #print(f'sorted_reuse_ratio {sorted_reuse_ratio[:10]}')

    # * the cache plan means what to materialize after processing a new batch
    if len(sorted_nodes)!=0:
      to_cache=sorted_nodes[:budget]
      cache_flag=np.zeros(num_embeddings)
      cache_flag[to_cache]=1
      cache_plan_list.append(to_cache)
    else:
      cache_plan_list.append(None)



  print(f'method reuse ratio, n_reuse {n_reuse}, n_recompute {n_recompute}')
  return cache_plan_list
'''