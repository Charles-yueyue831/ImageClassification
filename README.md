# ImageClassification
## image-classification-cifar10

This is your go-to-playground for training CNNs and Vision Transformers (ViT) and its related model on CIFAR-10, a common benchmark dataset in computer vision.

The whole codebase is implemented in PyTorch, which makes it easier for you to tweak and experiment.

### Updates

* Added WRN (Wide ResNet Networks)! (2024/04)

paper: Wide Residual Networks https://doi.org/10.48550/arXiv.1605.07146

![test_dataset accuracy.png](https://github.com/Charles-yueyue831/ImageClassification/blob/main/_WRN/image/test_dataset%20accuracy.png?raw=true)

### Usage example

1. WRN

python ./_WRN/main.py

### Results

|      | Test Accuracy | Train Log |
| ---- | ------------- | --------- |
| WRN  | 96.13%        |           |

