import torchvision.datasets as datasets
import torchvision
import torch

import MetaAugment.child_networks as cn
import MetaAugment.autoaugment_learners as aal

from .util_04_22 import *


# aa_learner config
config = {
        'sp_num' : 3,
        'learning_rate' : 1e-1,
        'toy_flag' : False,
#         'toy_flag' : True,
#         'toy_size' : 0.001,
        'batch_size' : 32,
        'max_epochs' : 100,
        'early_stop_num' : 10,
        }


# FashionMNIST with SimpleNet
train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=None)
test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                        train=False, download=True,
                        transform=torchvision.transforms.ToTensor())
child_network_architecture = cn.SimpleNet


# rs
run_benchmark(
    save_file='./benchmark/pickles/04_22_fm_sn_rs.pkl',
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    child_network_architecture=child_network_architecture,
    agent_arch=aal.randomsearch_learner,
    config=config,
    )