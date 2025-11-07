# Beyond Surface Facts: Deep Graph Augmentation and Wavelet Transform for Temporal Knowledge
Graph Completion
## Installation

Create a conda environment with pytorch and scikit-learn :

```
conda create --name tkbc_env python=3.7
source activate tkbc_env
conda install --file requirements.txt -c pytorch
```

Then install the tkbc package to this environment
```
python setup.py install
```

## Reproducing results

Run the following commands  to reproduce the results

```
python learner.py --dataset ICEWS14 --model DeepWave --rank 384 --emb_reg 1e-1 --time_reg 1e-4 --x 0.7 --y 0.8

python learner.py --dataset ICEWS05-15 --model DeepWave --rank 384 --emb_reg 1e-2 --time_reg 1e-2 --x 1.0 --y 0.9

python learner.py --dataset yago15k --model DeepWave --rank 384 --no_time_emb --emb_reg 1e-1 --time_reg 1e-1 --x 0.8 --y 0.9
```

