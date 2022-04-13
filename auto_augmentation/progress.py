from flask import Blueprint, request, render_template, flash, send_file
import subprocess

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
from MetaAugment import UCB1_JC  

bp = Blueprint("progress", __name__)

@bp.route("/user_input", methods=["GET", "POST"])
def response():

    # hyperparameters to change
    batch_size = 32       # size of batch the inner NN is trained with
    learning_rate = 1e-1  # fix learning rate
    ds = request.args["dataset_selection"]        # pick dataset (MNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100)
    toy_size = 0.02       # total propeortion of training and test set we use
    max_epochs = 100      # max number of epochs that is run if early stopping is not hit
    early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
    num_policies = 5      # fix number of policies
    num_sub_policies = 5  # fix number of sub-policies in a policy
    iterations = 100      # total iterations, should be more than the number of policies
    IsLeNet = request.args["network_selection"]   # using LeNet or EasyNet or SimpleNet ->> default 

    print(f'@@@@@ dataset is: {ds}, network is :{IsLeNet}')

    # generate random policies at start
    policies = UCB1_JC.generate_policies(num_policies, num_sub_policies)

    q_values, best_q_values = UCB1_JC.run_UCB1(policies, batch_size, learning_rate, ds, toy_size, max_epochs, early_stop_num, iterations, IsLeNet)

    plt.plot(best_q_values)

    best_q_values = np.array(best_q_values)
    save('best_q_values_{}_{}percent_{}.npy'.format(IsLeNet, int(toy_size*100), ds), best_q_values)
    #best_q_values = load('best_q_values_{}_{}percent_{}.npy'.format(IsLeNet, int(toy_size*100), ds), allow_pickle=True)


    return render_template("progress.html")