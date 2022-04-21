from dataclasses import dataclass
from flask import Flask, request
# from flask_cors import CORS

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

from ..library.MetaAugment import UCB1_JC_py as UCB1_JC
from ..library.MetaAugment import Evo_learner as Evo

app = Flask(__name__)


@app.route('/home', methods=["GET", "POST"])
def home():
    print('in flask home')
    form_data = request.get_json()
    batch_size = 1       # size of batch the inner NN is trained with
    learning_rate = 1e-1  # fix learning rate
    ds = form_data['select_dataset']      # pick dataset (MNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100)
    toy_size = form_data['toy_size']      # total propeortion of training and test set we use
    max_epochs = 10      # max number of epochs that is run if early stopping is not hit
    early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
    num_policies = 5      # fix number of policies
    num_sub_policies = 5  # fix number of sub-policies in a policy
    iterations = 5      # total iterations, should be more than the number of policies
    IsLeNet = form_data["network_selection"]   # using LeNet or EasyNet or SimpleNet ->> default 
    
    # if user upload datasets and networks, save them in the database
    if ds == 'Other':
        ds_folder = request.files['dataset_upload']
        ds_name_zip = ds_folder.filename
        ds_name = ds_name_zip.split('.')[0]
        ds_folder.save('./MetaAugment/datasets/'+ ds_name_zip)
        with zipfile.ZipFile('./MetaAugment/datasets/'+ ds_name_zip, 'r') as zip_ref:
            zip_ref.extractall('./MetaAugment/datasets/upload_dataset/')
        os.remove(f'./MetaAugment/datasets/{ds_name_zip}')

    else: 
        ds_name = None

    for (dirpath, dirnames, filenames) in os.walk(f'./MetaAugment/datasets/upload_dataset/{ds_name}/'):
        for dirname in dirnames:
            if dirname[0:6] != 'class_':
                return render_template("fail_dataset.html")
            else:
                pass


    if IsLeNet == 'Other':
        childnetwork = request.files['network_upload']
        childnetwork.save('./MetaAugment/child_networks/'+childnetwork.filename)

    
    # generate random policies at start
    auto_aug_leanrer = request.form.get("auto_aug_selection")

    if auto_aug_leanrer == 'UCB':
        policies = UCB1_JC.generate_policies(num_policies, num_sub_policies)
        q_values, best_q_values = UCB1_JC.run_UCB1(policies, batch_size, learning_rate, ds, toy_size, max_epochs, early_stop_num, iterations, IsLeNet, ds_name)
    elif auto_aug_leanrer == 'Evolutionary Learner':
        learner = Evo.Evolutionary_learner(fun_num=num_funcs, p_bins=1, mag_bins=1, sub_num_pol=1, ds_name=ds_name, exclude_method=exclude_method)
        learner.run_instance()
    elif auto_aug_leanrer == 'Random Searcher':
        pass 
    elif auto_aug_leanrer == 'Genetic Learner':
        pass

    plt.figure()
    plt.plot(q_values)


    best_q_values = np.array(best_q_values)

    # save('best_q_values_{}_{}percent_{}.npy'.format(IsLeNet, int(toy_size*100), ds), best_q_values)
    # best_q_values = load('best_q_values_{}_{}percent_{}.npy'.format(IsLeNet, int(toy_size*100), ds), allow_pickle=True)

    print("DONE")
    

    return None

@app.route('/api')
def index():
    return {'name': 'Hello'}