# General Description
- This is a PyTorch implementation of the paper "Discovering Posterior Label Hierarchy: Relaxed Tree-Wasserterin Distance meets Semi-supervised Learning (H-VLL)". More results to be updated...


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
cd CIFAR10
python train.py --dataset cifar10 --num-labeled 40 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 0 --out results/cifar10@40.0
```

## Experiments on STL-10 Dataset

### Requirements

Please see `STL-10/environment.yml`.

### Train H-VLL + Fixmatch
Train the model by 40 labeled data of CIFAR10 dataset:

```
cd STL-10
python fixmatch.py --c config/fixmatch/fixmatch_stl10_40_0.yaml
python fixmatch.py --c config/fixmatch/fixmatch_stl10_250_0.yaml
```

### Train H-VLL + Flexmatch
Train the model by 40 labeled data of CIFAR10 dataset:

```
cd STL-10
python flexmatch.py --c config/flexmatch/flexmatch_stl10_40_0.yaml
python flexmatch.py --c config/flexmatch/flexmatch_stl10_250_0.yaml
```


## References
- [Unofficial PyTorch implementation of FixMatch] (https://github.com/kekmodel/FixMatch-pytorch)

- [TorchSSL] (https://github.com/TorchSSL/TorchSSL)
