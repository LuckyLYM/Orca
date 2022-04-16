import math
import random
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size, strategy = None, ratio= None, reuse=True,cache_plan=None,use_partition=False,marker=False):

  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = batch_size
    num_test_instance = data.n_interactions
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    if strategy=='partition' and use_partition:
      num_test_batch = data.n_batch
    else:
      num_test_batch = math.ceil(num_test_instance/TEST_BATCH_SIZE)
 
    for batch_idx in range(num_test_batch):
      start_idx = batch_idx * TEST_BATCH_SIZE
      end_idx = min(num_test_instance, start_idx + TEST_BATCH_SIZE)
      sample_inds=np.array(list(range(start_idx,end_idx)))

      if strategy=='partition' and use_partition:
          batch_inds=data.tbatch[batch_idx]
          sample_size=int(len(batch_inds)*ratio)
          sample_inds=random.sample(batch_inds,sample_size)
      else:
          batch_size=end_idx-start_idx
          sample_size=int(batch_size*ratio)
          sample_inds=random.sample(range(start_idx,end_idx),sample_size)
          
      sources_batch = data.sources[sample_inds]
      destinations_batch = data.destinations[sample_inds]
      timestamps_batch = data.timestamps[sample_inds]
      edge_idxs_batch = data.edge_idxs[sample_inds]

      cache = None
      if cache_plan is not None:
        cache = cache_plan[batch_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)
      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,negative_samples, timestamps_batch,edge_idxs_batch, n_neighbors,reuse,cache,marker)
      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])
      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)






def eval_edge_prediction_tbatch(model, negative_edge_sampler, data, n_neighbors):
  
  # * Ensures the random sampler uses a seed for evaluation (i.e. we sample 
  # * always the same negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    num_test_instance = len(data.sources)
    num_test_batch = data.n_batch

    for k in range(num_test_batch):
      batch_inds=data.tbatch[k]
      sources_batch = data.sources[batch_inds]
      destinations_batch = data.destinations[batch_inds]
      timestamps_batch = data.timestamps[batch_inds]
      edge_idxs_batch = data.edge_idxs[batch_inds]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)
      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)
      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)




def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)
  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)
      
      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
