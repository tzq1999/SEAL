# General Description
- This is a PyTorch implementation of the paper "XMatch: Simultaneous Label Hierarchy Exploration And Semi-supervised Learning". We provide the code for CIFAR10, CIFAR100 and STL-10. Other settings are coming soon.


## Experiments on CIFAR10 Dataset

### Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)

### Train SEAL + DebiasPL
Train the model by 40 labeled data of CIFAR10 dataset:

```
cd CIFAR10
python train.py --dataset cifar10 --num-labeled 40 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 0 --out results/cifar10@40.0
```

## Experiments on CIFAR100 Dataset

### Requirements

Please see `CIFAR100/environment.yml`.

### Train SEAL + Fixmatch
Train the model by 400 labeled data of CIFAR100 dataset:

```
cd CIFAR100
python fixmatch.py --c config/fixmatch/fixmatch_cifar100_400_0.yaml
python fixmatch.py --c config/fixmatch/fixmatch_cifar100_2500_0.yaml
```

### Train SEAL + Flexmatch
Train the model by 400 labeled data of CIFAR100 dataset:

```
cd CIFAR100
python flexmatch.py --c config/flexmatch/flexmatch_cifar100_400_0.yaml
python flexmatch.py --c config/flexmatch/flexmatch_cifar100_2500_0.yaml
```

## Experiments on STL-10 Dataset

### Requirements

Please see `STL-10/environment.yml`.

### Train SEAL + Fixmatch
Train the model by 40 labeled data of STL-10 dataset:

```
cd STL-10
python fixmatch.py --c config/fixmatch/fixmatch_stl10_40_0.yaml
python fixmatch.py --c config/fixmatch/fixmatch_stl10_250_0.yaml
```

### Train SEAL + Flexmatch
Train the model by 40 labeled data of STL-10 dataset:

```
cd STL-10
python flexmatch.py --c config/flexmatch/flexmatch_stl10_40_0.yaml
python flexmatch.py --c config/flexmatch/flexmatch_stl10_250_0.yaml
```


## References
- [Unofficial PyTorch implementation of FixMatch] (https://github.com/kekmodel/FixMatch-pytorch)

- [TorchSSL] (https://github.com/TorchSSL/TorchSSL)
