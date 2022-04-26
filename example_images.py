#%%
import torchvision.datasets as datasets
import torchvision
import torch


from pathlib import Path

import math
import torch

from enum import Enum
from torch import Tensor
from typing import List, Tuple, Optional, Dict

from torchvision.transforms import functional as F, InterpolationMode

import matplotlib.pyplot as plt
from MetaAugment.main import *
import MetaAugment.child_networks as cn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F, InterpolationMode


# transform = transforms.Compose([
#                                                 F.adjust_saturation(1.0 + 3),
#                                                 F.adjust_contrast(4),
#                                                 transforms.ToTensor()
#                                             ])


train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                        train=False, download=True,
                        transform=torchvision.transforms.ToTensor())

#%%

for i, img in enumerate(train_dataset):
    # print("orig img: ")
    # plt.imshow(img[0].reshape((28, 28)), cmap='gray')
    # img = F.solarize(img[0], 8)
    # print("after solarise img: ", plt.imshow(img[0].reshape((28, 28)), cmap='gray'))

    # img = img[0].reshape((28, 28))
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # img = F.adjust_saturation(img, 1.0 + 3)
    # img = F.solarize(img, 8)

    # plt.figure()
    # plt.imshow(img, cmap='gray')
    img = img[0]
    print("first")
    plt.imshow(img.reshape((28, 28)))

    img = F.adjust_saturation(img, 1.0 + 3)
    print("second")
    plt.imshow(img.reshape((28, 28)))

    img = F.solarize(img, 8)
    print("third")
    plt.imshow(img.reshape((28, 28)))

    if i == 0:
        break

# %%

# %%
