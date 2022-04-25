import MetaAugment.autoaugment_learners as aal
import MetaAugment.child_networks as cn
import torch
import torchvision
import torchvision.datasets as datasets

import random

def test_generate_new_policy():
    """
    make sure randomsearch_learner.generate_new_policy() is robust
    with respect to different values of sp_num, fun_num, 
    p_bins, and m_bins
    """

    def my_test(discrete_p_m):
        for _ in range(40):
            sp_num = random.randint(1,20)
            
            p_bins = random.randint(2, 15)
            m_bins = random.randint(2, 15)

            agent = aal.randomsearch_learner(
                sp_num=sp_num,
                p_bins=p_bins,
                m_bins=m_bins,
                discrete_p_m=discrete_p_m
                )
            for _ in range(4):
                new_policy = agent.generate_new_policy()
                assert isinstance(new_policy, list), new_policy
    
    discrete_p_m = True
    my_test(discrete_p_m)

    discrete_p_m = False
    my_test(discrete_p_m)


def test_learn():
    """
    tests the randomsearch_learner.learn() method
    """
    train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=None)
    test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                            train=False, download=True,
                            transform=torchvision.transforms.ToTensor())
    child_network_architecture = cn.lenet
    # child_network_architecture = cn.lenet()

    agent = aal.randomsearch_learner(
                        sp_num=7,
                        toy_size=0.001,
                        batch_size=32,
                        learning_rate=0.05,
                        max_epochs=100,
                        early_stop_num=10,
                        )
    agent.learn(train_dataset,
                test_dataset,
                child_network_architecture=child_network_architecture,
                iterations=2)
