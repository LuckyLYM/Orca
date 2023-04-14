import numpy as np
from numba.experimental import jitclass
from numba import typed
import numba as nb
import torch
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)

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



def get_neighbor_finder(data, max_node_idx=None):
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


  def find_before(self, src_idx, cut_time):
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]


  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors):

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors),dtype=np.int32)  
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors),dtype=np.float32) 
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors),dtype=np.int32)

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,timestamp)  
      if len(source_neighbors) > 0 and n_neighbors > 0:
        source_edge_times = source_edge_times[-n_neighbors:]
        source_neighbors = source_neighbors[-n_neighbors:]
        source_edge_idxs = source_edge_idxs[-n_neighbors:]
        n_ngh=len(source_neighbors)
        neighbors[i, n_neighbors - n_ngh:] = source_neighbors
        edge_idxs[i, n_neighbors - n_ngh:] = source_edge_idxs
        edge_times[i, n_neighbors - n_ngh:] = source_edge_times
    return neighbors, edge_idxs, edge_times