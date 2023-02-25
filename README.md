Orca: Scalable Temporal Graph Neural Network Training with Theoretical Guarantees
=============================================================================

## Dataset
6 datasets were used in this paper:

- Wikipedia: downloadable from http://snap.stanford.edu/jodie/.
- Reddit: downloadable from http://snap.stanford.edu/jodie/.
- MOOC: downloadable from http://snap.stanford.edu/jodie/.
- AskUbuntu: downloadable from http://snap.stanford.edu/data/sx-askubuntu.html.
- SuperUser: downloadable from http://snap.stanford.edu/data/sx-superuser.html.
- Wiki-Talk: downloadable from http://snap.stanford.edu/data/wiki-talk-temporal.html.

## Preprocessing
If edge features or nodes features are absent, they will be replaced by a vector of zeros. Example usage:
```sh
python utils/preprocess_data.py --data wikipedia --bipartite
python uitls/preprocess_custom_data.py --data superuser
```

## Requirements
- PyTorch 1.7.1
- Python 3.8
- Numba 0.54.1

## Usage
```sh
Optional arguments:
    --data                  Dataset name
    --bs                    Batch size
    --n_degree              Number of neighbors to sample
    --n_head                Number of heads used in attention layer
    --n_epoch               Number of epochs
    --n_layer               Number of network layers
    --lr                    Learning rate
    --gpu                   GPU id
    --patience              Patience for early stopping
    --enable_random         Use random seeds
    --gradient              Disable gradient blocking
    --reuse                 Enable caching and reuse
    --budget                Cache size

    
Example usage:
    python train.py --n_epoch 50 --n_layer 2 --bs 200 -d wikipedia  --enable_random --reuse --budget 1000 --gpu 1
```
