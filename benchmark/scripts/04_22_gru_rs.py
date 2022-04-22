import torchvision.datasets as datasets
import torchvision
import torch

import MetaAugment.child_networks as cn
import MetaAugment.autoaugment_learners as aal

from pathlib import Path

"""
testing gru_learner and randomsearch_learner on

  fashionmnist with simple net

 and 

  cifar10 with lenet

"""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def run_benchmark(
    save_file,
    total_iter,
    train_dataset,
    test_dataset,
    child_network_architecture,
    agent_arch,
    config,
    ):
    try:
        # try to load agent
        with open(save_file, 'rb') as f:
            agent = torch.load(f, map_location=device)
    except FileNotFoundError:
        # if agent hasn't been saved yet, initialize the agent
        agent = agent_arch(**config)


    # if history is not length total_iter yet(if total_iter
    # different policies haven't been tested yet), keep running
    while len(agent.history)<total_iter:
        print(f'{len(agent.history)} / {total_iter}')
        # run 1 iteration (test one new policy and update the GRU)
        agent.learn(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    child_network_architecture=child_network_architecture,
                    iterations=1
                    )
        # save agent every iteration
        with open(save_file, 'wb+') as f:
            torch.save(agent, f)

    print('run_benchmark closing')


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
total_iter=150


# FashionMNIST with SimpleNet
train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=None)
test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                        train=False, download=True,
                        transform=torchvision.transforms.ToTensor())
child_network_architecture = cn.SimpleNet


# gru
run_benchmark(
    save_file='./benchmark/pickles/04_22_fm_sn_gru.pkl',
    total_iter=total_iter,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    child_network_architecture=child_network_architecture,
    agent_arch=aal.gru_learner,
    config=config,
    )

# rs
run_benchmark(
    save_file='./benchmark/pickles/04_22_fm_sn_rs.pkl',
    total_iter=total_iter,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    child_network_architecture=child_network_architecture,
    agent_arch=aal.randomsearch_learner,
    config=config,
    )


# CIFAR10 with LeNet
train_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=True, download=True, transform=None)
test_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=False, download=True, 
                        transform=torchvision.transforms.ToTensor())
child_network_architecture = cn.SimpleNet


# gru
run_benchmark(
    save_file='./benchmark/pickles/04_22_cf_ln_gru',
    total_iter=total_iter,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    child_network_architecture=child_network_architecture,
    agent_arch=aal.gru_learner,
    config=config,
    )

# rs
run_benchmark(
    save_file='./benchmark/pickles/04_22_cf_ln_rs',
    total_iter=total_iter,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    child_network_architecture=child_network_architecture,
    agent_arch=aal.randomsearch_learner,
    config=config,
    )
