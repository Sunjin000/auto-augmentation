from dataclasses import dataclass
from flask import Flask, request, current_app, render_template
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

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# # import agents and its functions
from MetaAugment.autoaugment_learners import ucb_learner as UCB1_JC
from MetaAugment.autoaugment_learners import evo_learner
import MetaAugment.controller_networks as cn
import MetaAugment.autoaugment_learners as aal
print('@@@ import successful')

# import agents and its functions
# from ..MetaAugment import UCB1_JC_py as UCB1_JC
# from ..MetaAugment import Evo_learner as Evo
# print('@@@ import successful')

app = Flask(__name__)


# it is used to collect user input and store them in the app
@app.route('/home', methods=["GET", "POST"])
def get_form_data():
    print('@@@ in Flask Home')
    # form_data = request.get_json() 
    # form_data = request.files['ds_upload'] 
    # print('@@@ form_data', form_data) 
 
    # form_data = request.form.get('test') 
    # print('@@@ this is form data', request.get_data())

    # required input
    # ds = form_data['select_dataset'] # pick dataset (MNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100)
    # IsLeNet = form_data["select_network"]   # using LeNet or EasyNet or SimpleNet ->> default 
    # auto_aug_learner = form_data["select_learner"] # augmentation methods to be excluded

    # print('@@@ required user input:', 'ds', ds, 'IsLeNet:', IsLeNet, 'auto_aug_leanrer:',auto_aug_learner)
    # # advanced input
    # if 'batch_size' in form_data.keys(): 
    #     batch_size = form_data['batch_size']       # size of batch the inner NN is trained with
    # else: 
    #     batch_size = 1 # this is for demonstration purposes
    # if 'learning_rate' in form_data.keys(): 
    #     learning_rate =  form_data['learning_rate']  # fix learning rate
    # else: 
    #     learning_rate = 10-1
    # if 'toy_size' in form_data.keys(): 
    #     toy_size = form_data['toy_size']      # total propeortion of training and test set we use
    # else: 
    #     toy_size = 1 # this is for demonstration purposes
    # if 'iterations' in form_data.keys(): 
    #     iterations = form_data['iterations']      # total iterations, should be more than the number of policies
    # else: 
    #     iterations = 10
    # exclude_method = form_data['select_action']
    # num_funcs = 14 - len(exclude_method)
    # print('@@@ advanced search: batch_size:', batch_size, 'learning_rate:', learning_rate, 'toy_size:', toy_size, 'iterations:', iterations, 'exclude_method', exclude_method, 'num_funcs', num_funcs)
    

    # # default values 
    # max_epochs = 10      # max number of epochs that is run if early stopping is not hit
    # early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
    # num_policies = 5      # fix number of policies
    # num_sub_policies = 5  # fix number of sub-policies in a policy
    
    
    # # if user upload datasets and networks, save them in the database
    # if ds == 'Other':
    #     ds_folder = request.files['ds_upload'] 
    #     print('!!!ds_folder', ds_folder)
    #     ds_name_zip = ds_folder.filename
    #     ds_name = ds_name_zip.split('.')[0]
    #     ds_folder.save('./datasets/'+ ds_name_zip)
    #     with zipfile.ZipFile('./datasets/'+ ds_name_zip, 'r') as zip_ref:
    #         zip_ref.extractall('./datasets/upload_dataset/')
    #     if not current_app.debug:
    #         os.remove(f'./datasets/{ds_name_zip}')
    # else: 
    #     ds_name = None

    # # test if uploaded dataset meets the criteria 
    # for (dirpath, dirnames, filenames) in os.walk(f'./datasets/upload_dataset/{ds_name}/'):
    #     for dirname in dirnames:
    #         if dirname[0:6] != 'class_':
    #             return None # neet to change render to a 'failed dataset webpage'

    # # save the user uploaded network
    # if IsLeNet == 'Other':
    #     childnetwork = request.files['network_upload']
    #     childnetwork.save('./child_networks/'+childnetwork.filename)
    #     network_name = childnetwork.filename

    
    # # generate random policies at start
    # current_app.config['AAL'] = auto_aug_learner
    # current_app.config['NP'] = num_policies
    # current_app.config['NSP'] = num_sub_policies
    # current_app.config['BS'] = batch_size
    # current_app.config['LR'] = learning_rate
    # current_app.config['TS'] = toy_size
    # current_app.config['ME'] = max_epochs
    # current_app.config['ESN'] = early_stop_num
    # current_app.config['IT'] = iterations
    # current_app.config['ISLENET'] = IsLeNet
    # current_app.config['DSN'] = ds_name
    # current_app.config['ds'] = ds

    
    # print("@@@ user input has all stored in the app")

    # data = {'ds': ds, 'ds_name': ds_name, 'IsLeNet': IsLeNet, 'ds_folder.filename': ds_name,
    #         'auto_aug_learner':auto_aug_learner, 'batch_size': batch_size, 'learning_rate': learning_rate, 
    #         'toy_size':toy_size, 'iterations':iterations, }
    
    # print('@@@ all data sent', data)
    return {'data': 'show training data'}

@app.route('/confirm', methods=['POST', 'GET'])
def confirm():
    print('inside confirm')
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

    data = {'ds': ds, 'ds_name': ds_name, 'IsLeNet': IsLeNet, 'ds_folder.filename': ds_name,
            'auto_aug_learner':auto_aug_learner, 'batch_size': batch_size, 'learning_rate': learning_rate, 
            'toy_size':toy_size, 'iterations':iterations, }
    return {'batch_size': '12'}

# ========================================================================
@app.route('/training', methods=['POST', 'GET'])
def training():
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
        q_values, best_q_values = UCB1_JC.run_UCB1(
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

        network = cn.evo_controller.evo_controller(fun_num=num_funcs, p_bins=1, m_bins=1, sub_num_pol=1)
        child_network = aal.evo.LeNet()
        learner = aal.evo.evo_learner(
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
        pass 
    elif auto_aug_learner == 'Genetic Learner':
        pass

    return {'status': 'training'}



# ========================================================================
@app.route('/results')
def show_result():
    return {'status': 'results'}

@app.route('/api')
def index():
    return {'status': 'api test'}


if __name__ == '__main__':
    app.run(debug=True)