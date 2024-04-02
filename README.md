# ImageClassification
## image-classification-cifar10

This is your go-to-playground for training CNNs and Vision Transformers (ViT) and its related model on CIFAR-10, a common benchmark dataset in computer vision.

The whole codebase is implemented in PyTorch, which makes it easier for you to tweak and experiment.

### Updates

* Added WRN (Wide ResNet Networks)! (2024/04)

paper: Wide Residual Networks https://doi.org/10.48550/arXiv.1605.07146

![WRN测试集测试结果.png](https://github.com/Charles-yueyue831/ImageClassification/blob/main/01%E5%AE%BD%E6%AE%8B%E5%B7%AE%E7%BD%91%E7%BB%9CWRN/image/WRN%E6%B5%8B%E8%AF%95%E9%9B%86%E6%B5%8B%E8%AF%95%E7%BB%93%E6%9E%9C.png?raw=true)

### Usage example

1. WRN

python ./_WRN/main.py

### Results

|      | Test Accuracy | Train Log |
| ---- | ------------- | --------- |
| WRN  | 96.13%        |           |

