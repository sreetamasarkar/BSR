from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import torch.distributed as dist
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

"""
mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
}t
"""


def Flowers_Dataset(args, split=None):
    """
    cifar 10
    """
    data_folder = '/data1/flowers'

    # mean = IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_DEFAULT_STD
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    
    train_set = datasets.Flowers102(root=data_folder,
                                      download=True,
                                      split='train',
                                      transform=train_transform)
    val_set = datasets.Flowers102(root=data_folder,
                                 download=True,
                                 split='val',
                                 transform=val_transform)
    test_set = datasets.Flowers102(root=data_folder,
                                 download=True,
                                 split='test',
                                 transform=val_transform)

    return train_set, val_set, test_set






    