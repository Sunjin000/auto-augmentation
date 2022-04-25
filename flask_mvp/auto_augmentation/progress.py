from flask import Blueprint, request, render_template, flash, send_file, current_app, g, session
import subprocess
import os
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.datasets as datasets

from matplotlib import pyplot as plt
from numpy import save, load
from tqdm import trange
torch.manual_seed(0)
# import agents and its functions

from MetaAugment.autoaugment_learners import ucb_learner
# hi
from MetaAugment import Evo_learner as Evo

import MetaAugment.autoaugment_learners as aal
from MetaAugment.main import create_toy
import MetaAugment.child_networks as cn
import pickle


bp = Blueprint("progress", __name__)


@bp.route("/user_input", methods=["GET", "POST"])
def response():

    # hyperparameters to change

    if request.method == 'POST':

        # generate random policies at start
        auto_aug_learner = request.form.get("auto_aug_selection")
        
        # search space & problem setting
        ds = request.form.get("dataset_selection")      # pick dataset (MNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100)
        ds_up = request.files['dataset_upload']
        exclude_method = request.form.getlist("action_space")
        num_funcs = 14 - len(exclude_method)
        num_policies = 5      # fix number of policies
        num_sub_policies = 5  # fix number of sub-policies in a policy
        toy_size = 1      # total propeortion of training and test set we use

        # child network
        IsLeNet = request.form.get("network_selection")   # using LeNet or EasyNet or SimpleNet ->> default 
        nw_up = childnetwork = request.files['network_upload']

        # child network training hyperparameters
        batch_size = 1       # size of batch the inner NN is trained with
        early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
        iterations = 5      # total iterations, should be more than the number of policies
        learning_rate = 1e-1  # fix learning rate
        max_epochs = 10      # max number of epochs that is run if early stopping is not hit

        # if user upload datasets and networks, save them in the database

        if ds == None and ds_up != None:
            ds = 'Other'
            ds_folder = request.files['dataset_upload']
            ds_name_zip = ds_folder.filename
            ds_name = ds_name_zip.split('.')[0]
            ds_folder.save('./MetaAugment/datasets/'+ ds_name_zip)
            with zipfile.ZipFile('./MetaAugment/datasets/'+ ds_name_zip, 'r') as zip_ref:
                zip_ref.extractall('./MetaAugment/datasets/upload_dataset/')
            if not current_app.debug:
                os.remove(f'./MetaAugment/datasets/{ds_name_zip}')


        else: 
            ds_name = None

        for (dirpath, dirnames, filenames) in os.walk(f'./MetaAugment/datasets/upload_dataset/{ds_name}/'):
            for dirname in dirnames:
                if dirname[0:6] != 'class_':
                    return render_template("fail_dataset.html")
                else:
                    pass


        if IsLeNet == None and nw_up != None:
            childnetwork = request.files['network_upload']
            childnetwork.save('./MetaAugment/child_networks/'+childnetwork.filename)
        


        if auto_aug_learner == 'UCB':
            policies = ucb_learner.generate_policies(num_policies, num_sub_policies)
            q_values, best_q_values = ucb_learner.run_UCB1(
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
            learner = Evo.Evolutionary_learner(
                                            fun_num=num_funcs, 
                                            p_bins=1, 
                                            mag_bins=1, 
                                            sub_num_pol=1, 
                                            ds_name=ds_name, 
                                            exclude_method=exclude_method
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
                                            toy_flag=True,
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

        plt.figure()
        plt.plot(q_values)


        # if auto_aug_learner == 'UCB':
        #     policies = ucb_learner.generate_policies(num_policies, num_sub_policies)
        #     q_values, best_q_values = ucb_learner.run_UCB1(policies, batch_size, learning_rate, ds, toy_size, max_epochs, early_stop_num, iterations, IsLeNet, ds_name)     
        #     # plt.figure()
        #     # plt.plot(q_values)
        #     best_q_values = np.array(best_q_values)

        # elif auto_aug_learner == 'Evolutionary Learner':
        #     network = Evo.Learner(fun_num=num_funcs, p_bins=1, m_bins=1, sub_num_pol=1)
        #     child_network = Evo.LeNet()
        #     learner = Evo.Evolutionary_learner(network=network, fun_num=num_funcs, p_bins=1, mag_bins=1, sub_num_pol=1, ds = ds, ds_name=ds_name, exclude_method=exclude_method, child_network=child_network)
        #     learner.run_instance()
        # elif auto_aug_learner == 'Random Searcher':
        #     pass 
        # elif auto_aug_learner == 'Genetic Learner':
        #     pass


    current_app.config['AAL'] = auto_aug_learner
    current_app.config['NP'] = num_policies
    current_app.config['NSP'] = num_sub_policies
    current_app.config['BS'] = batch_size
    current_app.config['LR'] = learning_rate
    current_app.config['TS'] = toy_size
    current_app.config['ME'] = max_epochs
    current_app.config['ESN'] = early_stop_num
    current_app.config['IT'] = iterations
    current_app.config['ISLENET'] = IsLeNet
    current_app.config['DSN'] = ds_name
    current_app.config['NUMFUN'] = num_funcs
    current_app.config['ds'] = ds
    current_app.config['exc_meth'] = exclude_method







    # return render_template("progress.html", exclude_method = exclude_method, auto_aug_learner=auto_aug_learner)
    return render_template("training.html", exclude_method = exclude_method, auto_aug_learner=auto_aug_learner)



