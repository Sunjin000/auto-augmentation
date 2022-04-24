from flask import Blueprint, request, render_template, flash, send_file, current_app
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

from MetaAugment.autoaugment_learners import ucb_learner as UCB1_JC
from MetaAugment import Evo_learner as Evo



bp = Blueprint("training", __name__)


@bp.route("/start_training", methods=["GET", "POST"])
def response():

    # hyperparameters to change

    # auto_aug_learner = session

    auto_aug_learner = current_app.config.get('AAL')

    num_policies = current_app.config.get('NP')
    num_sub_policies = current_app.config.get('NSP')
    batch_size = current_app.config.get('BS')
    learning_rate = current_app.config.get('LR')
    toy_size = current_app.config.get('TS')
    max_epochs = current_app.config.get('ME')
    early_stop_num = current_app.config.get('ESN')
    iterations = current_app.config.get('IT')
    IsLeNet = current_app.config.get('ISLENET')
    ds_name = current_app.config.get('DSN')
    num_funcs = current_app.config.get('NUMFUN')
    ds = current_app.config.get('ds')
    exclude_method = current_app.config.get('exc_meth')


    if auto_aug_learner == 'UCB':
        policies = UCB1_JC.generate_policies(num_policies, num_sub_policies)
        q_values, best_q_values = UCB1_JC.run_UCB1(policies, batch_size, learning_rate, ds, toy_size, max_epochs, early_stop_num, iterations, IsLeNet, ds_name)     
        best_q_values = np.array(best_q_values)

    elif auto_aug_learner == 'Evolutionary Learner':
        network = Evo.Learner(fun_num=num_funcs, p_bins=1, m_bins=1, sub_num_pol=1)
        child_network = Evo.LeNet()
        learner = Evo.Evolutionary_learner(network=network, fun_num=num_funcs, p_bins=1, mag_bins=1, sub_num_pol=1, ds = ds, ds_name=ds_name, exclude_method=exclude_method, child_network=child_network)
        learner.run_instance()
    elif auto_aug_learner == 'Random Searcher':
        pass 
    elif auto_aug_learner == 'Genetic Learner':
        pass

    return render_template("progress.html", auto_aug_learner=auto_aug_learner)




