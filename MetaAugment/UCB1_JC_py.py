#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.covariance import log_likelihood
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision
import torchvision.datasets as datasets
import pickle

from matplotlib import pyplot as plt
from numpy import save, load
from tqdm import trange

from .child_networks import *
from .main import create_toy, train_child_network


# In[6]:


"""Randomly generate 10 policies"""
"""Each policy has 5 sub-policies"""
"""For each sub-policy, pick 2 transformations, 2 probabilities and 2 magnitudes"""

def generate_policies(num_policies, num_sub_policies):
    
    policies = np.zeros([num_policies,num_sub_policies,6])

    # Policies array will be 10x5x6
    for policy in range(num_policies):
        for sub_policy in range(num_sub_policies):
            # pick two sub_policy transformations (0=rotate, 1=shear, 2=scale)
            policies[policy, sub_policy, 0] = np.random.randint(0,3)
            policies[policy, sub_policy, 1] = np.random.randint(0,3)
            while policies[policy, sub_policy, 0] == policies[policy, sub_policy, 1]:
                policies[policy, sub_policy, 1] = np.random.randint(0,3)

            # pick probabilities
            policies[policy, sub_policy, 2] = np.random.randint(0,11) / 10
            policies[policy, sub_policy, 3] = np.random.randint(0,11) / 10

            # pick magnitudes
            for transformation in range(2):
                if policies[policy, sub_policy, transformation] <= 1:
                    policies[policy, sub_policy, transformation + 4] = np.random.randint(-4,5)*5
                elif policies[policy, sub_policy, transformation] == 2:
                    policies[policy, sub_policy, transformation + 4] = np.random.randint(5,15)/10

    return policies


# In[7]:


"""Pick policy and sub-policy"""
"""Each row of data should have a different sub-policy but for now, this will do"""

def sample_sub_policy(policies, policy, num_sub_policies):
    sub_policy = np.random.randint(0,num_sub_policies)

    degrees = 0
    shear = 0
    scale = 1

    # check for rotations
    if policies[policy, sub_policy][0] == 0:
        if np.random.uniform() < policies[policy, sub_policy][2]:
            degrees = policies[policy, sub_policy][4]
    elif policies[policy, sub_policy][1] == 0:
        if np.random.uniform() < policies[policy, sub_policy][3]:
            degrees = policies[policy, sub_policy][5]

    # check for shears
    if policies[policy, sub_policy][0] == 1:
        if np.random.uniform() < policies[policy, sub_policy][2]:
            shear = policies[policy, sub_policy][4]
    elif policies[policy, sub_policy][1] == 1:
        if np.random.uniform() < policies[policy, sub_policy][3]:
            shear = policies[policy, sub_policy][5]

    # check for scales
    if policies[policy, sub_policy][0] == 2:
        if np.random.uniform() < policies[policy, sub_policy][2]:
            scale = policies[policy, sub_policy][4]
    elif policies[policy, sub_policy][1] == 2:
        if np.random.uniform() < policies[policy, sub_policy][3]:
            scale = policies[policy, sub_policy][5]

    return degrees, shear, scale


# In[8]:


"""Sample policy, open and apply above transformations"""
def run_UCB1(policies, batch_size, learning_rate, ds, toy_size, max_epochs, early_stop_num, early_stop_flag, average_validation, iterations, IsLeNet):

    # get number of policies and sub-policies
    num_policies = len(policies)
    num_sub_policies = len(policies[0])

    #Initialize vector weights, counts and regret
    q_values = [0]*num_policies
    cnts = [0]*num_policies
    q_plus_cnt = [0]*num_policies
    total_count = 0

    best_q_values = []

    for policy in trange(iterations):

        # get the action to try (either initially in order or using best q_plus_cnt value)
        if policy >= num_policies:
            this_policy = np.argmax(q_plus_cnt)
        else:
            this_policy = policy

        # get info of transformation for this sub-policy
        degrees, shear, scale = sample_sub_policy(policies, this_policy, num_sub_policies)

        # create transformations using above info
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomAffine(degrees=(degrees,degrees), shear=(shear,shear), scale=(scale,scale)),
            torchvision.transforms.ToTensor()])

        # open data and apply these transformations
        if ds == "MNIST":
            train_dataset = datasets.MNIST(root='./MetaAugment/train', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='./MetaAugment/test', train=False, download=True, transform=transform)
        elif ds == "KMNIST":
            train_dataset = datasets.KMNIST(root='./MetaAugment/train', train=True, download=True, transform=transform)
            test_dataset = datasets.KMNIST(root='./MetaAugment/test', train=False, download=True, transform=transform)
        elif ds == "FashionMNIST":
            train_dataset = datasets.FashionMNIST(root='./MetaAugment/train', train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(root='./MetaAugment/test', train=False, download=True, transform=transform)
        elif ds == "CIFAR10":
            train_dataset = datasets.CIFAR10(root='./MetaAugment/train', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='./MetaAugment/test', train=False, download=True, transform=transform)
        elif ds == "CIFAR100":
            train_dataset = datasets.CIFAR100(root='./MetaAugment/train', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(root='./MetaAugment/test', train=False, download=True, transform=transform)

        # check sizes of images
        img_height = len(train_dataset[0][0][0])
        img_width = len(train_dataset[0][0][0][0])
        img_channels = len(train_dataset[0][0])

        # check output labels
        if ds == "CIFAR10" or ds == "CIFAR100":
            num_labels = (max(train_dataset.targets) - min(train_dataset.targets) + 1)
        else:
            num_labels = (max(train_dataset.targets) - min(train_dataset.targets) + 1).item()

        # create toy dataset from above uploaded data
        train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size, toy_size)

        # create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if IsLeNet == "LeNet":
            model = LeNet(img_height, img_width, num_labels, img_channels).to(device) # added .to(device)
        elif IsLeNet == "EasyNet":
            model = EasyNet(img_height, img_width, num_labels, img_channels).to(device) # added .to(device)
        else:
            model = SimpleNet(img_height, img_width, num_labels, img_channels).to(device) # added .to(device)
        sgd = optim.SGD(model.parameters(), lr=1e-1)
        cost = nn.CrossEntropyLoss()

        # set variables for best validation accuracy and early stop count
        best_acc = 0
        early_stop_cnt = 0
        total_val = 0

        # train model and check validation accuracy each epoch
        for _epoch in range(max_epochs):

            # train model
            model.train()
            for idx, (train_x, train_label) in enumerate(train_loader):
                train_x, train_label = train_x.to(device), train_label.to(device) # new code
                label_np = np.zeros((train_label.shape[0], num_labels))
                sgd.zero_grad()
                predict_y = model(train_x.float())
                loss = cost(predict_y, train_label.long())
                loss.backward()
                sgd.step()

            # check validation accuracy on validation set
            correct = 0
            _sum = 0
            model.eval()
            for idx, (test_x, test_label) in enumerate(test_loader):
                test_x, test_label = test_x.to(device), test_label.to(device) # new code
                predict_y = model(test_x.float()).detach()
                #predict_ys = np.argmax(predict_y, axis=-1)
                predict_ys = torch.argmax(predict_y, axis=-1) # changed np to torch
                #label_np = test_label.numpy()
                _ = predict_ys == test_label
                #correct += np.sum(_.numpy(), axis=-1)
                correct += np.sum(_.cpu().numpy(), axis=-1) # added .cpu()
                _sum += _.shape[0]
            
            acc = correct / _sum

            if average_validation[0] <= _epoch <= average_validation[1]:
                total_val += acc

            # update best validation accuracy if it was higher, otherwise increase early stop count
            if acc > best_acc :
                best_acc = acc
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            # exit if validation gets worse over 10 runs and using early stopping
            if early_stop_cnt >= early_stop_num and early_stop_flag:
                break

            # exit if using fixed epoch length
            if _epoch >= average_validation[1] and not early_stop_flag:
                best_acc = total_val / (average_validation[1] - average_validation[0] + 1)
                break

        # update q_values
        if policy < num_policies:
            q_values[this_policy] += best_acc
        else:
            q_values[this_policy] = (q_values[this_policy]*cnts[this_policy] + best_acc) / (cnts[this_policy] + 1)

        best_q_value = max(q_values)
        best_q_values.append(best_q_value)

        if (policy+1) % 10 == 0:
            print("Iteration: {},\tQ-Values: {}, Best Policy: {}".format(policy+1, list(np.around(np.array(q_values),2)), max(list(np.around(np.array(q_values),2)))))

        # update counts
        cnts[this_policy] += 1
        total_count += 1

        # update q_plus_cnt values every turn after the initial sweep through
        if policy >= num_policies - 1:
            for i in range(num_policies):
                q_plus_cnt[i] = q_values[i] + np.sqrt(2*np.log(total_count)/cnts[i])

    return q_values, best_q_values


# # In[9]:

if __name__=='__main__':
    batch_size = 32       # size of batch the inner NN is trained with
    learning_rate = 1e-1  # fix learning rate
    ds = "MNIST"          # pick dataset (MNIST, KMNIST, FashionMNIST, CIFAR10, CIFAR100)
    toy_size = 0.02       # total propeortion of training and test set we use
    max_epochs = 100      # max number of epochs that is run if early stopping is not hit
    early_stop_num = 10   # max number of worse validation scores before early stopping is triggered
    early_stop_flag = True        # implement early stopping or not
    average_validation = [15,25]  # if not implementing early stopping, what epochs are we averaging over
    num_policies = 5      # fix number of policies
    num_sub_policies = 5  # fix number of sub-policies in a policy
    iterations = 100      # total iterations, should be more than the number of policies
    IsLeNet = "SimpleNet" # using LeNet or EasyNet or SimpleNet

    # generate random policies at start
    policies = generate_policies(num_policies, num_sub_policies)

    q_values, best_q_values = run_UCB1(policies, batch_size, learning_rate, ds, toy_size, max_epochs, early_stop_num, early_stop_flag, average_validation, iterations, IsLeNet)

    plt.plot(best_q_values)

    best_q_values = np.array(best_q_values)
    save('best_q_values_{}_{}percent_{}.npy'.format(IsLeNet, int(toy_size*100), ds), best_q_values)
    #best_q_values = load('best_q_values_{}_{}percent_{}.npy'.format(IsLeNet, int(toy_size*100), ds), allow_pickle=True)

