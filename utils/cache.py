import numpy as np
import math 
import time
import numba as nb
from numba import jit, typed
import sys

@nb.njit(parallel=True)
def isin(a, b):
    out=np.empty(a.shape[0], dtype=nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        if a[i] in b:
            out[i]=True
        else:
            out[i]=False
    return out

def read_batches(args,train_data,neighbor_finder,num_embeddings):
  BATCH_SIZE = args.bs
  n_degree=args.n_degree
  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance/BATCH_SIZE)

  total_n_in=0
  total_n_unique_in=0
  total_n_out=0
  total_n_unique_out=0

  target_list = typed.List()
  ngh_list = typed.List()
  occur_list = typed.List()
  
  for _ in range(num_embeddings):
    empty_list=typed.List()
    empty_list.append((0,0))
    occur_list.append(empty_list)

  for batch_idx in range(0, num_batch):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(num_instance, start_idx + BATCH_SIZE)
    sample_inds=np.array(list(range(start_idx,end_idx)))

    sources_batch, destinations_batch = train_data.sources[sample_inds],train_data.destinations[sample_inds]
    timestamps_batch = train_data.timestamps[sample_inds]

    source_nodes = np.concatenate([sources_batch, destinations_batch])
    timestamps = np.concatenate([timestamps_batch, timestamps_batch])

    neighbors, _, _ = neighbor_finder.get_temporal_neighbor(source_nodes,timestamps,n_degree)
    neighbors=neighbors[neighbors!=0]  #[400,10] => 1 dimensional array

    unique_target=np.unique(source_nodes)
    unique_neighbors=np.unique(neighbors)


    # unique in-batch nerighbors
    unique_in = np.intersect1d(unique_target, unique_neighbors)
    in_index = np.isin(neighbors,unique_in)
    
    n_in = np.count_nonzero(in_index)
    n_unique_in = len(unique_in)
    total_n_in += n_in
    total_n_unique_in+=n_unique_in

    # index of those in-batch neighbors
    out_index = ~in_index
    out = neighbors[out_index]
    unique_out=np.unique(out)

    n_out= len(out)
    n_unique_out = len(unique_out)  
    total_n_out += n_out
    total_n_unique_out+=n_unique_out

    target_list.append(unique_target)   # unique target nodes
    ngh_list.append(out)                # not unique out-of-batch neighbors
    
    # * update occur list
    for target in unique_target:
      occur_list[target].append((batch_idx,0))    # * 0 means target node

    for ngh in unique_out:
      occur_list[ngh].append((batch_idx,1))       # * 1 means neighbor node

  for i in range(num_embeddings):                 # * 2 means end-of-the batch
    occur_list[i].append((num_batch,2))

  return num_batch,target_list,ngh_list,occur_list,total_n_in,total_n_unique_in,total_n_out,total_n_unique_out


@jit(nopython=True)
def MRU_numba(num_embeddings,num_batch,budget,target_list,ngh_list,occur_list):
  
  MAX_DISTANCE=100000000
  n_reuse=0
  n_recompute=0
  total_reuse_distance = 0

  cache_plan_list=[]
  cache_flag=np.zeros(num_embeddings)
  time_flag=np.zeros(num_embeddings)

  # index starts from 1 because of the we have padded dummpy nodes in the occur_list
  index_list= np.ones(num_embeddings,np.int32)    

  for batch_idx in range(num_batch):

    target=target_list[batch_idx]
    ngh=ngh_list[batch_idx] # not unique because we want to compute n_recompute and n_reuse
    
    cache_=cache_flag[ngh]
    index=np.where(cache_==0)[0]
    uncached_ngh=ngh[index] 
    n_recompute+=len(uncached_ngh)
    
    index=np.where(cache_==1)[0]
    cached_ngh=ngh[index]
    n_reuse+=len(cached_ngh)

    # reuse distance of cache neighbors
    total_reuse_distance+= np.sum(batch_idx - time_flag[cached_ngh])
    cached=np.where(cache_flag==1)[0]                       # * unique newly computed nodes
    new_computed = np.concatenate((uncached_ngh,target))
    new_computed = np.unique(new_computed)
    candidates=np.concatenate((uncached_ngh,cached,target)) # * unique candidates nodes
    candidates=np.unique(candidates)

    reuse_distance_list=[]
    for node in candidates:  
      occur_index=index_list[node]
      occurs=occur_list[node]
      while occurs[occur_index][0]<=batch_idx: 
        occur_index+=1

      state=occurs[occur_index]
      id = state[0]
      value = state[1]
      index_list[node]=occur_index
      
      if value == 0:    # target node
        reuse_distance_list.append(MAX_DISTANCE+1)
      elif value ==2:   # not appear anymore
        reuse_distance_list.append(MAX_DISTANCE+1)
      else:             # neighbor node
        reuse_distance_list.append(id-batch_idx)


    # sort by the next reuse time
    reuse_distance_list=np.array(reuse_distance_list)
    sorted_inds=np.argsort(reuse_distance_list)
    sorted_nodes=candidates[sorted_inds]

    to_cache=sorted_nodes[:budget]
    cache_flag=np.zeros(num_embeddings)
    cache_flag[to_cache]=1
    cache_plan_list.append(to_cache)

    # update the time flag, which is used for computing reuse distance
    new_index = isin(to_cache,new_computed) 
    new_nodes = to_cache[new_index]
    time_flag[new_nodes]=batch_idx

  
  average_reuse_distance = total_reuse_distance/n_reuse
  return cache_plan_list, n_reuse, n_recompute, average_reuse_distance







############################################# 2Q #############################################
@jit(nopython=True)
def TwoQ_numba(num_embeddings,num_batch,budget,target_list,ngh_list):

  half_budget = budget//2

  n_FIF_cached=0
  n_LRU_cached=0
  n_reuse=0
  n_recompute=0
  total_reuse_distance = 0
  
  cache_plan_list=[]
  LRU_cache_flag=np.zeros(num_embeddings)
  LRU_time_flag=np.zeros(num_embeddings)
  LRU_arrive_time=np.zeros(num_embeddings)

  FIF_cache_flag=np.zeros(num_embeddings)
  FIF_time_flag=np.zeros(num_embeddings)
  FIF_arrive_time=np.zeros(num_embeddings)


  for batch_idx in range(num_batch):
    ngh=ngh_list[batch_idx]   

    ######### FIF part #########
    FIF_cache_=FIF_cache_flag[ngh]

    # FIF cached neighbors => move to LRU           # not unique
    index=np.where(FIF_cache_==1)[0]
    FIF_cached_nghs=ngh[index]
    n_reuse+=len(FIF_cached_nghs)                   # increment counter
    total_reuse_distance+= np.sum(batch_idx - FIF_arrive_time[FIF_cached_nghs])


    nodes_FIF_to_LRU = np.unique(FIF_cached_nghs)   # unique FIF cached neighbors
    n_FIF_to_LRU=len(nodes_FIF_to_LRU)

    # FIF uncached neighbors
    index=np.where(FIF_cache_==0)[0]                # not unique
    FIF_uncached_nghs=ngh[index] 
    

    ######### LRU part #########
    LRU_cache_=LRU_cache_flag[FIF_uncached_nghs]

    # LRU uncached neighbors => added to FIF
    index=np.where(LRU_cache_==0)[0]                # not unique
    LRU_uncached_nghs=FIF_uncached_nghs[index] 
    n_recompute+=len(LRU_uncached_nghs)             # increment counter

    nodes_new_to_FIF=np.unique(LRU_uncached_nghs)   # unique new to FIF neighbors
    n_new_to_FIF = len(nodes_new_to_FIF)
  

    # LRU cached neighbors => update time flag
    index=np.where(LRU_cache_==1)[0]
    LRU_cached_nghs=FIF_uncached_nghs[index]
    n_reuse+=len(LRU_cached_nghs)                   # increment counter
    total_reuse_distance+= np.sum(batch_idx - LRU_arrive_time[LRU_cached_nghs])

    nodes_LRU_to_LRU=np.unique(LRU_cached_nghs)     # unique LRU to LRU neighbors
    n_LRU_to_LRU=len(nodes_LRU_to_LRU)

    ########### update FIF ###########
    n_FIF_used = n_FIF_cached-n_FIF_to_LRU
    n_FIF_available = half_budget -n_FIF_used


    FIF_cache_flag[nodes_FIF_to_LRU]=0
    if n_new_to_FIF<=n_FIF_available:    # no need to evict from FIF
      FIF_cache_flag[nodes_new_to_FIF]=1
      FIF_time_flag[nodes_new_to_FIF]=batch_idx
      FIF_arrive_time[nodes_new_to_FIF]=batch_idx
      n_FIF_cached=n_FIF_used+n_new_to_FIF

    elif n_new_to_FIF>=half_budget:      # evict all and selective cache
      nodes_new_selected_to_FIF=np.random.choice(nodes_new_to_FIF,half_budget,replace=False)
      FIF_cache_flag=np.zeros(num_embeddings)
      FIF_time_flag=np.zeros(num_embeddings)
      FIF_arrive_time=np.zeros(num_embeddings)

      FIF_cache_flag[nodes_new_selected_to_FIF]=1
      FIF_time_flag[nodes_new_selected_to_FIF]=batch_idx
      FIF_arrive_time[nodes_new_selected_to_FIF]=batch_idx
      n_FIF_cached=half_budget

    else:                                # selective eviction
      n_FIF_evict = n_new_to_FIF-n_FIF_available
      nodes_FIF_remained=np.where(FIF_cache_flag==1)[0]

      # selective evict
      nodes_FIF_evict=np.random.choice(nodes_FIF_remained,n_FIF_evict,replace=False) 
      FIF_cache_flag[nodes_FIF_evict]=0

      # cache all the new
      FIF_cache_flag[nodes_new_to_FIF]=1
      FIF_time_flag[nodes_new_to_FIF]=batch_idx
      FIF_arrive_time[nodes_new_to_FIF]=batch_idx
      n_FIF_cached=half_budget


    ########### update LRU ###########
    LRU_time_flag[nodes_LRU_to_LRU]=batch_idx
    n_LRU_available = half_budget - n_LRU_cached
    n_LRU_may_evict = n_LRU_cached - n_LRU_to_LRU

    if n_FIF_to_LRU<=n_LRU_available:   # no evict

      LRU_cache_flag[nodes_FIF_to_LRU]=1
      LRU_time_flag[nodes_FIF_to_LRU]=batch_idx
      LRU_arrive_time[nodes_FIF_to_LRU]=FIF_arrive_time[nodes_FIF_to_LRU]
      n_LRU_cached=n_LRU_cached+n_FIF_to_LRU

    elif n_FIF_to_LRU<=(n_LRU_available+n_LRU_may_evict):  # selective evict
      n_LRU_evict = n_FIF_to_LRU - n_LRU_available

      # selective eviction
      LRU_cached = np.where(LRU_cache_flag==1)[0]
      last_time=LRU_time_flag[LRU_cached]
      evicted_inds=np.argsort(last_time)[:n_LRU_evict]
      evicted_nodes = LRU_cached[evicted_inds]
      LRU_cache_flag[evicted_nodes]=0

      # cache all the new
      LRU_cache_flag[nodes_FIF_to_LRU]=1
      LRU_time_flag[nodes_FIF_to_LRU]=batch_idx
      LRU_arrive_time[nodes_FIF_to_LRU]=FIF_arrive_time[nodes_FIF_to_LRU]
      n_LRU_cached=half_budget

    else:  # evict all and selective push
      
      # evict all
      LRU_cached = np.where(LRU_cache_flag==1)[0]
      last_time=LRU_time_flag[LRU_cached]
      evicted_inds = np.where(last_time<batch_idx)[0]
      evicted_nodes = LRU_cached[evicted_inds]
      LRU_cache_flag[evicted_nodes]=0

      # selective cache
      n_FIF_selected_to_LRU = half_budget-n_LRU_to_LRU
      nodes_FIF_selected_to_LRU=np.random.choice(nodes_FIF_to_LRU,n_FIF_selected_to_LRU,replace=False)  
      LRU_cache_flag[nodes_FIF_selected_to_LRU]=1
      LRU_time_flag[nodes_FIF_selected_to_LRU]=batch_idx
      LRU_arrive_time[nodes_FIF_selected_to_LRU]=FIF_arrive_time[nodes_FIF_selected_to_LRU]
      n_LRU_cached=half_budget


    ### get cache plan
    nodes_FIF_cached = np.where(FIF_cache_flag==1)[0]
    nodes_LRU_cached = np.where(LRU_cache_flag==1)[0]
    
    n_FIF_cached=len(nodes_FIF_cached)
    n_LRU_cached=len(nodes_LRU_cached)

    nodes_cached = np.hstack((nodes_FIF_cached,nodes_LRU_cached))
    cache_plan_list.append(nodes_cached)

  average_reuse_distance = total_reuse_distance/n_reuse
  return cache_plan_list, n_reuse, n_recompute, average_reuse_distance



############################################# LRU #############################################
@jit(nopython=True)
def LRU_numba(num_embeddings,num_batch,budget,target_list,ngh_list):

  n_cached=0
  n_reuse=0
  n_recompute=0
  total_reuse_distance = 0
  cache_plan_list=[]
  
  cache_flag=np.zeros(num_embeddings)   
  time_flag=np.zeros(num_embeddings)
  arrive_time=np.zeros(num_embeddings)
  

  for batch_idx in range(num_batch):
    ngh=ngh_list[batch_idx]   
    target=ngh_list[batch_idx]

    cache_=cache_flag[ngh]
    
    # uncached neighbors
    index=np.where(cache_==0)[0]    # not unique here
    uncached_nghs=ngh[index] 
    n_recompute+=len(uncached_nghs)
    
    # cached neighbors
    index=np.where(cache_==1)[0]    # not unique here
    cached_nghs=ngh[index]
    n_reuse+=len(cached_nghs)
    total_reuse_distance+= np.sum(batch_idx - arrive_time[cached_nghs])

    # update the reuse timestamp here
    time_flag[cached_nghs]=batch_idx

    # index of cached nodes
    cached_nodes=np.where(cache_flag==1)[0]
    n_cached=len(cached_nodes)
    

    # LRU will not query and cache the newly computed nodes
    candidates = np.unique(uncached_nghs) 
    candidate_size = len(candidates)

    available_size = budget-n_cached
    evict_size = 0 if available_size>=candidate_size else min(candidate_size-available_size,n_cached)
    to_cache_size = candidate_size if (evict_size+available_size)>=candidate_size else budget


    # evict some items here
    if evict_size!=0:
      last_time=time_flag[cached_nodes]
      sorted_inds=np.argsort(last_time)[:evict_size]
      evicted_nodes = cached_nodes[sorted_inds]
      cache_flag[evicted_nodes]=0
    
    # cache some new items here
    to_cache=np.random.choice(candidates,to_cache_size,replace=False)
    cache_flag[to_cache]=1
    time_flag[to_cache]=batch_idx
    arrive_time[to_cache]=batch_idx

    # update the cache_plan_list
    cache_plan_list.append(np.where(cache_flag==1)[0])
  
  average_reuse_distance = total_reuse_distance/n_reuse
  return cache_plan_list, n_reuse, n_recompute, average_reuse_distance



def get_cache_plan(args,train_data,neighbor_finder,num_embeddings,strategy):
  
  budget=args.budget
  prepare_start=time.time()

  num_batch,target_list,ngh_list,occur_list,total_n_in,total_n_unique_in,total_n_out,total_n_unique_out=read_batches(args,train_data,neighbor_finder,num_embeddings)
  t_prepare=time.time()-prepare_start

  start=time.time()
  if strategy=='MRU':
    plan,n_reuse,n_recompute,average_reuse_distance=MRU_numba(num_embeddings,num_batch,budget,target_list,ngh_list,occur_list)
  elif strategy=='LRU':
    plan,n_reuse,n_recompute,average_reuse_distance=LRU_numba(num_embeddings,num_batch,budget,target_list,ngh_list)
  elif strategy=='2Q':
    plan,n_reuse,n_recompute,average_reuse_distance=TwoQ_numba(num_embeddings,num_batch,budget,target_list,ngh_list)
  else:
    print(f'unsupported cache strategy {strategy}')
    sys.exit(1)
  t_cache=time.time()-start

  print(f'prepare {t_prepare}, cache {t_cache}, n_reuse {n_reuse}, n_recompute {n_recompute}, average_reuse_distance {average_reuse_distance}')
  return plan
