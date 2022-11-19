# General Description
- This is a PyTorch implementation of the paper "Discovering Posterior Label Hierarchy: Relaxed Tree-Wasserterin Distance meets Semi-supervised Learning (H-VLL)". We only release the results for DebiasPL on CIFAR10 and. Other settings are coming soon.


## Experiments on CIFAR10 Dataset

### Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)

### Train H-VLL + DebiasPL
Train the model by 40 labeled data of CIFAR10 dataset:

```
python train.py --dataset cifar10 --num-labeled 40 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 0 --out results/cifar10@40.0
```



## Experiments on CIFAR100 Dataset

### Requirements

Please see `CIFAR100/environment.yml`.

## Experiments on STL-10 Dataset

### Requirements

Please see `STL-10/environment.yml`.


## References
- [Unofficial PyTorch implementation of FixMatch] (https://github.com/kekmodel/FixMatch-pytorch)

- [TorchSSL] (https://github.com/TorchSSL/TorchSSL)
