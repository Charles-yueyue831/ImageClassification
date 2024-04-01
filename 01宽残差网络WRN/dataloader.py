# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : dataloader.py
# @Software : PyCharm

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np


def get_loader(batch_size, num_workers):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=r'E:\python_dataset\cifar-10', train=True, download=True,
                                                 transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=r'E:\python_dataset\cifar-10', train=False, download=True,
                                                transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, drop_last=True, )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                                              shuffle=False, pin_memory=True, drop_last=False, )
    return train_loader, test_loader