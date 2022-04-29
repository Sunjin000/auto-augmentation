import torchvision.datasets as datasets
import torchvision
import torch

import autoaug.child_networks as cn
import autoaug.autoaugment_learners as aal

from .util_04_22 import *


# AaLearner config
config = {
        'sp_num' : 3,
        'learning_rate' : 1e-1,
#         'toy_size' : 0.001,
        'batch_size' : 32,
        'max_epochs' : 100,
        'early_stop_num' : 10,
        }


# CIFAR10 with LeNet
train_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=True, download=True, transform=None)
test_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=False, download=True, 
                        transform=torchvision.transforms.ToTensor())
child_network_architecture = cn.LeNet(
                                    img_height=32,
                                    img_width=32,
                                    num_labels=10,
                                    img_channels=3
                                    )


save_dir='./benchmark/pickles/04_22_cf_ln_gru'

# rs
run_benchmark(
    save_file=save_dir+'.pkl',
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    child_network_architecture=child_network_architecture,
    agent_arch=aal.GruLearner,
    config=config,
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