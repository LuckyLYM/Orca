import numpy as np
import random
import pandas as pd

class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)
    self.tbatch = None
    self.n_batch = 0

  def sample(self,ratio):
    data_size=self.n_interactions
    sample_size=int(ratio*data_size)
    sample_inds=random.sample(range(data_size),sample_size)
    sample_inds=np.sort(sample_inds)
    sources=self.sources[sample_inds]
    destination=self.destinations[sample_inds]
    timestamps=self.timestamps[sample_inds]
    edge_idxs=self.edge_idxs[sample_inds]
    labels=self.labels[sample_inds]
    return Data(sources,destination,timestamps,edge_idxs,labels)


############## load a batch of training data ##############
def get_data(dataset_name):
  graph_df = pd.read_csv('../data/{}/ml_{}.csv'.format(dataset_name,dataset_name))
  edge_features = np.load('../data/{}/ml_{}.npy'.format(dataset_name,dataset_name))
  node_features = np.load('../data/{}/ml_{}_node.npy'.format(dataset_name,dataset_name)) 
    
  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values
  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
  
  random.seed(2020)
  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  test_node_set = set(sources[timestamps > val_time]).union(set(destinations[timestamps > val_time]))
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)
  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0


  # * the val set can indeed contain the new test node
  new_node_set = node_set - train_node_set
  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)

  test_mask = timestamps > test_time
  edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
  new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
  new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])
  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])
  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask])


  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

  return node_features, edge_features, full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data

