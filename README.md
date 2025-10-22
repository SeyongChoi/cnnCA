# cnnCA

**cnnCA** is a deep learning framework for predicting the **contact angle** of liquid droplets on **rough or patterned surfaces**, based on **2D lattice (grid) representations** of surface morphology.  
It provides implementations of multiple neural architectures — including **Steerable CNN**, **CNN**, and **ANN** — to investigate how surface geometry and symmetry affect wetting behavior.

## Overview

The **cnnCA** project aims to develop and analyze deep learning architectures for predicting the **contact angle** of liquid droplets on **rough or patterned solid surfaces**, which plays a crucial role in understanding **wetting phenomena** and **surface wettability**.  
By leveraging 2D grid-based surface data (height profiles or roughness maps), the model learns to infer the resulting macroscopic contact angle through supervised learning.

The project provides a modular framework that supports:
- **ANN (Artificial Neural Network)** for baseline regression
- **CNN (Convolutional Neural Network)** for spatial feature extraction
- **Steerable CNN** for incorporating **rotational equivariance**, enabling physically consistent predictions under rotation and symmetry transformations

Training, validation, and evaluation workflows are fully configurable via `.yaml` files, allowing users to easily adjust hyperparameters, data normalization, and logging options.  
All experiments are implemented in **PyTorch** and **PyTorch Lightning**, with **Weights & Biases (wandb)** and **TensorBoard** integration for monitoring and visualization.

-----
**Status:** *In progress* 
- [ ] Implement the SteearbleCNN module
-----

## Useage

### 1. Installation
```bash
# Conda 환경 예시
conda create -n cnnca python=3.8
conda activate cnnca

# 저장소 클론
git clone https://github.com/SeyongChoi/cnnCA.git
cd cnnCA

```

### 2. Prepare the setting input file (.yaml)

```yaml
# Load for dataset
dataset:
  data_root_dir: "D:\\SteerableCNNCA\\data\\"
  normalize:
    ca_int: True
    height: True
  # Setting for the dataset
  grid_size: 100 
  pbc_step: 15
  split: [0.7, 0.1, 0.2] #[train_ratio, val_ratio, test_ratio]

# Model settings
model:
  type: "ANN" # "ANN" or "CNN" or "SteerableCNN"
  # hyperparameters for Model
  # For commonly used in ANN, CNN, SteerableCNN
  hidden_dims: [1000, 100]  # Fully Connected Layer의 hidden layer
  dropout_rates: [0.2, 0.2] # Fully Connected Layer의 dropout rate 
  weight_init: "he_normal"  # 'xavier_uniform' or 'xavier_normal' or 'he_uniform' or 'he_normal' or 'kaiming_uniform' or 'kaiming_normal' or 'orthogonal' or 'default'
  
# setting for the training
training:
  device: "cpu"                # "cpu" or "cuda"
  batch_size: 1
  max_epochs: 10
  lr: 0.001                    # learning rate        
  loss_fn: "mse"               # "mse" or "mae"
  optimizer: "adam"            # "adam" or "sgd" or "rmsprop"
  l2_reg: 0.1                  # L2 regularization strength

# setting for the logging
logging:
    wandb:
      enable: True
      project_name: "SteerableCNNCA"
      run_name: "ann_run_1"                # Name of the run
    TensorBoard:
      enable: True
      log_dir: "logs"                  # Directory to save TensorBoard logs
```

### 3. Run the training the model

```bash
python main.py --config config/input.yaml
```

## Dependecy

- Python 3.8+
- PyTorch
- PyTorch Lightning
- [ESCNN](https://github.com/QUVA-Lab/escnn)
- wandb
- numpy, matplotlib, scikit-learn 등

