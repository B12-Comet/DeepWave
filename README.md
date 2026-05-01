# Beyond Surface Facts: Deep Graph Augmentation and Wavelet Transform for Temporal Knowledge Graph Completion

<img width="956" height="564" alt="e2500b865e427f822cb9e2bc03fffe63" src="https://github.com/user-attachments/assets/287dfd56-7bfb-43db-b762-7cbfa8adfc05" />



## Installation

Create a conda environment with pytorch and scikit-learn :

```
conda create --name tkbc_env python=3.7
source activate tkbc_env
conda install --file requirements.txt -c pytorch
```

### requirements

```
tqdm==4.67.1
pytorch==2.4.1
numpy==1.24.4
scikit-learn==1.3.2
scipy==1.10.0
pytorch_wavelets==1.3.0
```

Then install the tkbc package to this environment

```
python setup.py install
```

## Datasets

We use ICEWS14, ICEWS05-15, YAGO15k datasets for temporal knowledge graph link prediction.

## Results

The results are:

| Dataset    | MRR  | H@1  | H@3  | H@10  |
| ---------- | ---- | ---- | ---- | ---- |
| ICEWS14    | 97.9 | 97.3 | 98.4 | 98.9 |
| ICEWS05-15 | 97.3 | 97.2 | 97.4 | 97.6 |
| YAGO15k    | 86.3 | 85.1 | 87.0 | 88.5 |

## How to run

Run the following commands  to reproduce the results

```
python learner.py --dataset ICEWS14 --model DeepWave --rank 384 --emb_reg 1e-1 --time_reg 1e-4 --x 0.7 --y 0.8

python learner.py --dataset ICEWS05-15 --model DeepWave --rank 384 --emb_reg 1e-2 --time_reg 1e-2 --x 1.0 --y 0.9

python learner.py --dataset yago15k --model DeepWave --rank 384 --no_time_emb --emb_reg 1e-1 --time_reg 1e-1 --x 0.8 --y 0.9
```

