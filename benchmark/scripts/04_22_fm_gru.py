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


save_dir='./benchmark/pickles/04_22_fm_sn_gru'

# rs
run_benchmark(
    save_file=save_dir+'.pkl',
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    child_network_architecture=child_network_architecture,
    agent_arch=aal.gru_learner,
    config=config,
    total_iter=144
    )

rerun_best_policy(
    agent_pickle=save_dir+'.pkl',
    accs_txt=save_dir+'.txt',
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    child_network_architecture=child_network_architecture,
    config=config,
    repeat_num=5
    )