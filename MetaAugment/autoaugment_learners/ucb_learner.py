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
        
        super().__init__(sp_num=sp_num, 
                        fun_num=14,
                        p_bins=p_bins, 
                        m_bins=m_bins, 
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
        self.policies = []
        self.make_more_policies()

        # attributes used in the UCB1 algorithm
        self.q_values = [0]*self.num_policies
        self.best_q_values = []
        self.cnts = [0]*self.num_policies
        self.q_plus_cnt = [0]*self.num_policies
        self.total_count = 0



    def make_more_policies(self, n):
        """generates n more random policies and adds it to self.policies

        Args:
            n (int): how many more policies to we want to randomly generate
                    and add to our list of policies
        """

        self.policies.append([self.generate_new_policy() for _ in n])


    def learn(self, 
            train_dataset, 
            test_dataset, 
            child_network_architecture, 
            iterations=15):


        for this_iter in trange(iterations):

            # get the action to try (either initially in order or using best q_plus_cnt value)
            # TODO: change this if statemetn
            if this_iter >= self.num_policies:
                this_policy_idx = np.argmax(self.q_plus_cnt)
                this_policy = self.policies[this_policy_idx]
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
            # TODO: change this if statemetn
            if this_iter < self.num_policies:
                self.q_values[this_policy_idx] += best_acc
            else:
                self.q_values[this_policy_idx] = (self.q_values[this_policy_idx]*self.cnts[this_policy_idx] + best_acc) / (self.cnts[this_policy_idx] + 1)

            best_q_value = max(self.q_values)
            self.best_q_values.append(best_q_value)

            if (this_iter+1) % 5 == 0:
                print("Iteration: {},\tQ-Values: {}, Best this_iter: {}".format(
                                this_iter+1, 
                                list(np.around(np.array(self.q_values),2)), 
                                max(list(np.around(np.array(self.q_values),2)))
                                )
                    )

            # update counts
            self.cnts[this_policy_idx] += 1
            self.total_count += 1

            # update q_plus_cnt values every turn after the initial sweep through
            # TODO: change this if statemetn
            if this_iter >= self.num_policies - 1:
                for i in range(self.num_policies):
                    self.q_plus_cnt[i] = self.q_values[i] + np.sqrt(2*np.log(self.total_count)/self.cnts[i])

            


       




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