"""
CONTAINS THE FUNTIONS THAT THE WEBAPP CAN USE TO INTERACT WITH
THE LIBRARY
"""

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets

# # import agents and its functions
import MetaAugment.autoaugment_learners as aal
import MetaAugment.controller_networks as cont_n
import MetaAugment.child_networks as cn
from MetaAugment.main import create_toy

import pickle

def parse_users_learner_spec(
            # aalearner type
            auto_aug_learner, 
            # search space settings
            ds, 
            ds_name, 
            exclude_method, 
            num_funcs, 
            num_policies, 
            num_sub_policies, 
            # child network settings
            toy_size, 
            IsLeNet, 
            batch_size, 
            early_stop_num, 
            iterations, 
            learning_rate, 
            max_epochs
            ):
    """
    The website receives user inputs on what they want the aa_learner
    to be. We take those hyperparameters and return an aa_learner

    """
    if auto_aug_learner == 'UCB':
        policies = aal.ucb_learner.generate_policies(num_policies, num_sub_policies)
        q_values, best_q_values = aal.ucb_learner.run_UCB1(
                                                policies,
                                                batch_size, 
                                                learning_rate, 
                                                ds, 
                                                toy_size, 
                                                max_epochs, 
                                                early_stop_num, 
                                                iterations, 
                                                IsLeNet, 
                                                ds_name
                                                )     
        best_q_values = np.array(best_q_values)
    elif auto_aug_learner == 'Evolutionary Learner':
        network = cont_n.evo_controller(fun_num=num_funcs, p_bins=1, m_bins=1, sub_num_pol=1)
        child_network = cn.LeNet()
        learner = aal.evo_learner(
                                network=network, 
                                fun_num=num_funcs, 
                                p_bins=1, 
                                mag_bins=1, 
                                sub_num_pol=1, 
                                ds = ds, 
                                ds_name=ds_name, 
                                exclude_method=exclude_method, 
                                child_network=child_network
                                )
        learner.run_instance()
    elif auto_aug_learner == 'Random Searcher':
            # As opposed to when ucb==True, `ds` and `IsLenet` are processed outside of the agent
            # This system makes more sense for the user who is not using the webapp and is instead
            # using the library within their code
        download = True
        if ds == "MNIST":
            train_dataset = datasets.MNIST(root='./MetaAugment/datasets/mnist/train', train=True, download=download)
            test_dataset = datasets.MNIST(root='./MetaAugment/datasets/mnist/test', train=False,
                                                download=download, transform=torchvision.transforms.ToTensor())
        elif ds == "KMNIST":
            train_dataset = datasets.KMNIST(root='./MetaAugment/datasets/kmnist/train', train=True, download=download)
            test_dataset = datasets.KMNIST(root='./MetaAugment/datasets/kmnist/test', train=False,
                                                download=download, transform=torchvision.transforms.ToTensor())
        elif ds == "FashionMNIST":
            train_dataset = datasets.FashionMNIST(root='./MetaAugment/datasets/fashionmnist/train', train=True, download=download)
            test_dataset = datasets.FashionMNIST(root='./MetaAugment/datasets/fashionmnist/test', train=False,
                                                download=download, transform=torchvision.transforms.ToTensor())
        elif ds == "CIFAR10":
            train_dataset = datasets.CIFAR10(root='./MetaAugment/datasets/cifar10/train', train=True, download=download)
            test_dataset = datasets.CIFAR10(root='./MetaAugment/datasets/cifar10/test', train=False,
                                                download=download, transform=torchvision.transforms.ToTensor())
        elif ds == "CIFAR100":
            train_dataset = datasets.CIFAR100(root='./MetaAugment/datasets/cifar100/train', train=True, download=download)
            test_dataset = datasets.CIFAR100(root='./MetaAugment/datasets/cifar100/test', train=False,
                                                download=download, transform=torchvision.transforms.ToTensor())
        elif ds == 'Other':
            dataset = datasets.ImageFolder('./MetaAugment/datasets/'+ ds_name)
            len_train = int(0.8*len(dataset))
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len(dataset)-len_train])

            # check sizes of images
        img_height = len(train_dataset[0][0][0])
        img_width = len(train_dataset[0][0][0][0])
        img_channels = len(train_dataset[0][0])
            # check output labels
        if ds == 'Other':
            num_labels = len(dataset.class_to_idx)
        elif ds == "CIFAR10" or ds == "CIFAR100":
            num_labels = (max(train_dataset.targets) - min(train_dataset.targets) + 1)
        else:
            num_labels = (max(train_dataset.targets) - min(train_dataset.targets) + 1).item()
            # create toy dataset from above uploaded data
        train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size, toy_size)
            # create model
        if IsLeNet == "LeNet":
            model = cn.LeNet(img_height, img_width, num_labels, img_channels)
        elif IsLeNet == "EasyNet":
            model = cn.EasyNet(img_height, img_width, num_labels, img_channels)
        elif IsLeNet == 'SimpleNet':
            model = cn.SimpleNet(img_height, img_width, num_labels, img_channels)
        else:
            model = pickle.load(open(f'datasets/childnetwork', "rb"))

            # use an aa_learner. in this case, a rs learner
        agent = aal.randomsearch_learner(batch_size=batch_size,
                                            learning_rate=learning_rate,
                                            toy_size=toy_size,
                                            max_epochs=max_epochs,
                                            early_stop_num=early_stop_num,
                                            )
        agent.learn(train_dataset,
                        test_dataset,
                        child_network_architecture=model,
                        iterations=iterations)
    elif auto_aug_learner == 'Genetic Learner':
        pass