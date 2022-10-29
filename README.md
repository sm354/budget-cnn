# Image classification on a budget: Training classifiers progressively with a fixed time budget

This repository contains the code to reduce the training time of Convolutional Neural Networks (CNNs) by upto 50% without hurting the performance by more than 1%. In this work, we train a classifier, say ResNet18, with initially lesser number of parameters and on lower resolution images, and then progressively (over time or training epochs), we increase the number of parameters and image resolution by our algorithm. In a nutshell, our algorithm provides an initialisation strategy for the newly introduced weights (during up-sampling while training) using the weights that have been trained till that point.

## Results
We benchmark using ResNet18 over four datasets: Cifar-10, -100, FashionMNIST, and tiny ImageNet. We find that there is a consistent \~50% reduction in training time with utmost 1% drop in classifcation accuracy.

| Dataset      | Acc (baseline) | Acc (ours) | Train Time (baseline) | Train Time (ours) |
|--------------|----------------|------------|-----------------------|-------------------|
| CIFAR-10     | 94.2           | 93.5       | 7h 10m                | 3h 30m            |
| CIFAR-100    | 74.4           | 73.5       | 7h 9m                 | 3h 36m            |
| FashionMNIST | 95.2           | 95.2       | 6h 24m                | 3h 20m            |
| TinyImageNet | 61.9           | 61.8       | 23h 40m               | 10h 30m           |

### Installation
```
pip install -r requirements.txt
```

### Train ResNet18 in conventional way
```
python train.py --num_epochs 100 --dataset cifar10 --exp_name R18_normal
```


### Train ResNet18 using our algorithm
```
python train.py --algo yes --num_epochs 100 --dataset cifar10 --exp_name R18_normal
```

### Acknowledgement
This work was done during a research internship (May - July, 2020) under the guidance of [Prof. Ernest Chong](https://people.sutd.edu.sg/~ernest_chong/) at Singpore University of Technology & Design (SUTD).
