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

from MetaAugment import UCB1_JC_py as UCB1_JC
from MetaAugment import Evo_learner as Evo



bp = Blueprint("progress", __name__)


@bp.route("/user_input", methods=["GET", "POST"])
def response():

    # hyperparameters to change

    if request.method == 'POST':

        exclude_method = request.form.getlist("action_space")
        num_funcs = 14 - len(exclude_method)

        batch_size = 1       # size of batch the inner NN is trained with
        learning_rate = 1e-1  # fix learning rate
        ds = request.form.get("dataset_selection")      # pick dataset (MNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100)
        ds_up = request.files['dataset_upload']
        nw_up = childnetwork = request.files['network_upload']
        toy_size = 1      # total propeortion of training and test set we use
        max_epochs = 10      # max number of epochs that is run if early stopping is not hit
        early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
        num_policies = 5      # fix number of policies
        num_sub_policies = 5  # fix number of sub-policies in a policy
        iterations = 5      # total iterations, should be more than the number of policies
        IsLeNet = request.form.get("network_selection")   # using LeNet or EasyNet or SimpleNet ->> default 

        # if user upload datasets and networks, save them in the database

        # print("DS HERE3: ", ds)
        # print("ds up: ", ds_up)
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


        auto_aug_learner = request.form.get("auto_aug_selection")

        # if auto_aug_learner == 'UCB':
        #     policies = UCB1_JC.generate_policies(num_policies, num_sub_policies)
        #     q_values, best_q_values = UCB1_JC.run_UCB1(policies, batch_size, learning_rate, ds, toy_size, max_epochs, early_stop_num, iterations, IsLeNet, ds_name)     
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



