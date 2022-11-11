# PRISM
PyTorch implementation of PRISM: A Self-Supervised Learning Framework using Causal Mechanism for use in Single Domain Generalization Project. TY

Including support for:
- Distributed data parallel training
- Global batch normalization
- LARS (Layer-wise Adaptive Rate Scaling) optimizer.

- Reference Code (https://github.com/Spijkervet/SimCLR)


### Quickstart (fine-tune linear classifier)


### Training ResNet encoder:


### Distributed Training
With distributed data parallel (DDP) training:
```
CUDA_VISIBLE_DEVICES=0 python main.py --nodes 2 --nr 0
CUDA_VISIBLE_DEVICES=1 python main.py --nodes 2 --nr 1
CUDA_VISIBLE_DEVICES=2 python main.py --nodes 2 --nr 2
CUDA_VISIBLE_DEVICES=N python main.py --nodes 2 --nr 3
```


### Results




### Pre-trained models
| ResNet (batch_size, epochs) | Optimizer | STL-10 Top-1 |
| ------------- | ------------- | ------------- |
| ResNet18(64,100) | LARS       | [TODO]        |

#### LARS optimizer
The LARS optimizer is implemented in `modules/lars.py`. It can be activated by adjusting the `config/config.yaml` optimizer setting to: `optimizer: "LARS"`. It is still experimental and has not been thoroughly tested. (From https://github.com/Spijkervet/SimCLR)

## What is PRISM?



## Usage


## Configuration


## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir runs
```

## Optimizers and learning rate schedule

#### Dependencies
```
torch
torchvision
tensorboard
pyyaml
```
# PRISM
