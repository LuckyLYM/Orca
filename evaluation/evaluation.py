import math
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size, reuse=True,cache_plan=None):

  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc, val_acc = [], [], []
  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = batch_size
    num_test_instance = data.n_interactions
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
 
    for batch_idx in range(num_test_batch):
      start_idx = batch_idx * TEST_BATCH_SIZE
      end_idx = min(num_test_instance, start_idx + TEST_BATCH_SIZE)
      sample_inds=np.array(list(range(start_idx,end_idx)))

      sources_batch = data.sources[sample_inds]
      destinations_batch = data.destinations[sample_inds]
      timestamps_batch = data.timestamps[sample_inds]
      edge_idxs_batch = data.edge_idxs[sample_inds]

      cache = None
      if cache_plan is not None:
        cache = cache_plan[batch_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)
      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, n_neighbors,reuse, False, cache)
      
      pos_prob=pos_prob.cpu().numpy() 
      neg_prob=neg_prob.cpu().numpy()
      pred_score = np.concatenate([pos_prob, neg_prob]) 
      true_label = np.concatenate([np.ones(size), np.zeros(size)])  
      true_binary_label= np.zeros(size) 
      pred_binary_label = np.argmax(np.hstack([pos_prob,neg_prob]),axis=1) 

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
      val_acc.append(accuracy_score(true_binary_label, pred_binary_label))

  return np.mean(val_ap), np.mean(val_auc), np.mean(val_acc)
