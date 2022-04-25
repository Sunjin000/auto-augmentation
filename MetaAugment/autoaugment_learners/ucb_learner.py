#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import trange

from ..child_networks import *
from ..main import train_child_network
from .randomsearch_learner import randomsearch_learner
from .aa_learner import augmentation_space


class ucb_learner(randomsearch_learner):
    """
    Tests randomly sampled policies from the search space specified by the AutoAugment
    paper. Acts as a baseline for other aa_learner's.
    """
    def __init__(self,
                # parameters that define the search space
                sp_num=5,
                fun_num=14,
                p_bins=11,
                m_bins=10,
                discrete_p_m=True,
                # hyperparameters for when training the child_network
                batch_size=8,
                toy_flag=False,
                toy_size=0.1,
                learning_rate=1e-1,
                max_epochs=float('inf'),
                early_stop_num=30,
                # ucb_learner specific hyperparameter
                num_policies=100
                ):
        
        super().__init__(sp_num, 
                fun_num, 
                p_bins, 
                m_bins, 
                discrete_p_m=discrete_p_m,
                batch_size=batch_size,
                toy_flag=toy_flag,
                toy_size=toy_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                early_stop_num=early_stop_num,)
        
        self.num_policies = num_policies

        # When this learner is initialized we generate `num_policies` number
        # of random policies. 
        # generate_new_policy is inherited from the randomsearch_learner class
        self.policies = [self.generate_new_policy() for _ in self.num_policies]

        # attributes used in the UCB1 algorithm
        self.q_values = [0]*self.num_policies
        self.cnts = [0]*self.num_policies
        self.q_plus_cnt = [0]*self.num_policies
        self.total_count = 0

    def learn(self, 
            train_dataset, 
            test_dataset, 
            child_network_architecture, 
            iterations=15):

        #Initialize vector weights, counts and regret


        best_q_values = []

        for this_iter in trange(iterations):

            # get the action to try (either initially in order or using best q_plus_cnt value)
            if this_iter >= self.num_policies:
                this_policy = self.policies[np.argmax(self.q_plus_cnt)]
            else:
                this_policy = this_iter


            best_acc = self.test_autoaugment_policy(
                                this_policy,
                                child_network_architecture,
                                train_dataset,
                                test_dataset,
                                logging=False
                                )

            # update q_values
            if this_iter < self.num_policies:
                self.q_values[this_policy] += best_acc
            else:
                self.q_values[this_policy] = (self.q_values[this_policy]*self.cnts[this_policy] + best_acc) / (self.cnts[this_policy] + 1)

            best_q_value = max(self.q_values)
            best_q_values.append(best_q_value)

            if (this_iter+1) % 5 == 0:
                print("Iteration: {},\tQ-Values: {}, Best this_iter: {}".format(
                                this_iter+1, 
                                list(np.around(np.array(self.q_values),2)), 
                                max(list(np.around(np.array(self.q_values),2)))
                                )
                    )

            # update counts
            self.cnts[this_policy] += 1
            self.total_count += 1

            # update q_plus_cnt values every turn after the initial sweep through
            if this_iter >= self.num_policies - 1:
                for i in range(self.num_policies):
                    self.q_plus_cnt[i] = self.q_values[i] + np.sqrt(2*np.log(self.total_count)/self.cnts[i])

            # yield q_values, best_q_values
        return self.q_values, best_q_values


       

    
def run_UCB1(
            policies, 
            batch_size, 
            learning_rate, 
            ds, 
            toy_size, 
            max_epochs, 
            early_stop_num, 
            early_stop_flag, 
            average_validation, 
            iterations, 
            IsLeNet
        ):
    pass

def generate_policies(
            num_policies, 
            self.sp_num
        ):
    pass



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
    sp_num = 5  # fix number of sub-policies in a policy
    iterations = 100      # total iterations, should be more than the number of policies
    IsLeNet = "SimpleNet" # using LeNet or EasyNet or SimpleNet