# 📣 Drug-Drug-Synergy model (DDoS) 💊

This repository contains the code for the DDoS model implementation with PyTorch. 

DDoS is a Graph Neural Network (GNN) based neural network model used for drug-drug-cell line synergy predictions.


## Installation

* `git clone` the repo and `cd` into it.
* Run `pip install -e .` to install the repo's python package.

## Running 🏃

1. use `notebooks/DDoS_Preprocessing.ipynb` and `notebooks/DDoS_Dataset_Generation.ipynb` to generate DataTensors from the drug combination features (chemical structures and gene expression).
2. use `notebooks/DDoS-Hyperparam.ipynb` to find the best performing model hyperparameters.
3. use `notebooks/DDoS_Train_Eval.ipynb` to train / test on the best analyze the results.