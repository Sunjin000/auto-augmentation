from flask import Blueprint, request, render_template, flash, send_file
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
import MetaAugment.autoaugment_learners as aal
from MetaAugment.main import create_toy
from MetaAugment.child_networks import *
import pickle


bp = Blueprint("progress", __name__)


@bp.route("/user_input", methods=["GET", "POST"])
def response():

    # hyperparameters to change
    # print("thing: ", request.files['dataset_upload'] )
    if request.method == 'POST':
        batch_size = 1       # size of batch the inner NN is trained with
        learning_rate = 1e-1  # fix learning rate
        ds = request.form.get("dataset_selection")      # pick dataset (MNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100)
        toy_size = 1      # total propeortion of training and test set we use
        max_epochs = 10      # max number of epochs that is run if early stopping is not hit
        early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
        num_policies = 5      # fix number of policies
        num_sub_policies = 5  # fix number of sub-policies in a policy
        iterations = 5      # total iterations, should be more than the number of policies
        IsLeNet = request.form.get("network_selection")   # using LeNet or EasyNet or SimpleNet ->> default 

        print(f'@@@@@ dataset is: {ds}, network is :{IsLeNet}')

        # if user upload datasets and networks, save them in the database
        if ds == 'Other':
            ds_folder = request.files['dataset_upload']
            print('@@@ ds_folder', ds_folder)
            ds_name_zip = ds_folder.filename
            ds_folder.save('./MetaAugment/datasets/'+ ds_name_zip)
            with zipfile.ZipFile('./MetaAugment/datasets/'+ ds_name_zip, 'r') as zip_ref:
                zip_ref.extractall('./MetaAugment/datasets/')
            ds_name = ds_name_zip.split('.')[0]

        else: 
            ds_name = None

        
        if IsLeNet == 'Other':
            childnetwork = request.files['network_upload']
            childnetwork.save('./MetaAugment/child_networks/'+childnetwork.filename)


        ucb = True # I made this dummy variable so my commit does not change this file's
                   # behaviour
        if ucb==True:
            # generate random policies at start
            policies = UCB1_JC.generate_policies(num_policies, num_sub_policies)
            q_values, best_q_values = UCB1_JC.run_UCB1(policies, batch_size, learning_rate, ds, toy_size, max_epochs, early_stop_num, iterations, IsLeNet, ds_name)
            print("q_values: ", q_values)

            plt.figure()
            plt.plot(q_values)
            plt.savefig('/static/image/test.png')

            # plt.plot(best_q_values)

            best_q_values = np.array(best_q_values)
            # save('best_q_values_{}_{}percent_{}.npy'.format(IsLeNet, int(toy_size*100), ds), best_q_values)
            #best_q_values = load('best_q_values_{}_{}percent_{}.npy'.format(IsLeNet, int(toy_size*100), ds), allow_pickle=True)

        else: 
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
                train_dataset = datasets.CIFAR10(root='./MetaAugment/datasets/fashionmnist/train', train=True, download=download)
                test_dataset = datasets.CIFAR10(root='./MetaAugment/datasets/fashionmnist/test', train=False,
                                                download=download, transform=torchvision.transforms.ToTensor())
            elif ds == "CIFAR100":
                train_dataset = datasets.CIFAR100(root='./MetaAugment/datasets/fashionmnist/train', train=True, download=download)
                test_dataset = datasets.CIFAR100(root='./MetaAugment/datasets/fashionmnist/test', train=False,
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
                model = LeNet(img_height, img_width, num_labels, img_channels)
            elif IsLeNet == "EasyNet":
                model = EasyNet(img_height, img_width, num_labels, img_channels)
            elif IsLeNet == 'SimpleNet':
                model = SimpleNet(img_height, img_width, num_labels, img_channels)
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

        print("DONE")


    return render_template("progress.html")





########### TESTING STUFF

# UPLOAD_FOLDER = '/datasets'

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/user_input', methods = ['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             return redirect(url_for('uploaded_file', filename=filename))
#     return '''
    
#     '''