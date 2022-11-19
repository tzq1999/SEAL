# General Description
- This is a PyTorch implementation of the paper "Discovering Posterior Label Hierarchy: Relaxed Tree-Wasserterin Distance meets Semi-supervised Learning". We only release the results for DebiasPL on CIFAR10. Other settings are coming soon.



## Usage

### Train
Train the model by 40 labeled data of CIFAR-10 dataset:

```
python train.py --dataset cifar10 --num-labeled 40 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 0 --out results/cifar10@40.0
```






## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)


## References
- [Unofficial PyTorch implementation of FixMatch] (https://github.com/kekmodel/FixMatch-pytorch)
