import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision.transforms.autoaugment as autoaugment
import random
import pygad
import pygad.torchga as torchga
import random
import copy
from torchvision.transforms import functional as F, InterpolationMode
from typing import List, Tuple, Optional, Dict
import heapq



# from MetaAugment.main import *
# import MetaAugment.child_networks as child_networks


np.random.seed(0)
random.seed(0)


augmentation_space = [
            # (function_name, do_we_need_to_specify_magnitude)
            ("ShearX", True),
            ("ShearY", True),
            ("TranslateX", True),
            ("TranslateY", True),
            ("Rotate", True),
            ("Brightness", True),
            ("Color", True),
            ("Contrast", True),
            ("Sharpness", True),
            ("Posterize", True),
            ("Solarize", True),
            ("AutoContrast", False),
            ("Equalize", False),
            ("Invert", False),
        ]

class Learner(nn.Module):
    def __init__(self, fun_num=14, p_bins=11, m_bins=10, sub_num_pol=5):
        self.fun_num = fun_num
        self.p_bins = p_bins 
        self.m_bins = m_bins 
        self.sub_num_pol = sub_num_pol

        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, self.sub_num_pol * 2 * (self.fun_num + self.p_bins + self.m_bins))

# Currently using discrete outputs for the probabilities 

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)

        return y


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        return y



# code from https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/train.py
# def train_model(full_policy, child_network):
#     """
#     Takes in the specific transformation index and probability 
#     """

#     # transformation = generate_policy(5, ps, mags)

#     train_transform = transforms.Compose([
#                                             full_policy,
#                                             transforms.ToTensor()
#                                         ])

#     batch_size = 32
#     n_samples = 0.005

#     train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, download=False, transform=train_transform)
#     test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, download=False, transform=torchvision.transforms.ToTensor())

#     train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size, 0.01)


#     sgd = optim.SGD(child_network.parameters(), lr=1e-1)
#     cost = nn.CrossEntropyLoss()
#     epoch = 20


#     best_acc = train_child_network(child_network, train_loader, test_loader,
#                                      sgd, cost, max_epochs=100, print_every_epoch=False)

#     return best_acc





# ORGANISING DATA

# transforms = ['RandomResizedCrop', 'RandomHorizontalFlip', 'RandomVerticalCrop', 'RandomRotation']
train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, download=True, transform=torchvision.transforms.ToTensor())
n_samples = 0.02
# shuffle and take first n_samples  %age of training dataset
shuffled_train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)).tolist())
indices_train = torch.arange(int(n_samples*len(train_dataset)))
reduced_train_dataset = torch.utils.data.Subset(shuffled_train_dataset, indices_train)
# shuffle and take first n_samples %age of test dataset
shuffled_test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)).tolist())
indices_test = torch.arange(int(n_samples*len(test_dataset)))
reduced_test_dataset = torch.utils.data.Subset(shuffled_test_dataset, indices_test)

train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=60000)



class Evolutionary_learner():

    def __init__(self, network, num_solutions = 30, num_generations = 10, num_parents_mating = 15, train_loader = None, child_network = None, p_bins = 11, mag_bins = 10, sub_num_pol=5, fun_num = 14, augmentation_space = None, train_dataset = None, test_dataset = None):
        self.auto_aug_agent = Learner(fun_num=fun_num, p_bins=p_bins, m_bins=mag_bins, sub_num_pol=sub_num_pol)
        self.torch_ga = torchga.TorchGA(model=network, num_solutions=num_solutions)
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.initial_population = self.torch_ga.population_weights
        self.train_loader = train_loader
        self.child_network = child_network
        self.p_bins = p_bins 
        self.sub_num_pol = sub_num_pol
        self.mag_bins = mag_bins
        self.fun_num = fun_num
        self.augmentation_space = augmentation_space
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        assert num_solutions > num_parents_mating, 'Number of solutions must be larger than the number of parents mating!'

        self.set_up_instance()
    

    def generate_policy(self, sp_num, ps, mags):
        """
        
        """
        policies = []
        for subpol in range(sp_num):
            sub = []
            for idx in range(2):
                transformation = augmentation_space[(2*subpol) + idx]
                p = ps[(2*subpol) + idx]
                mag = mags[(2*subpol) + idx]
                sub.append((transformation, p, mag))
            policies.append(tuple(sub))
        
        return policies

# Every image has specific operation. Policy for every image (2 (trans., prob., mag) output)


# RNN -> change the end -/- leave for now, ask Javier


# Use mini-batch with current output, get mode transformation -> mean probability and magnitude
#   Pass through each image in mini-batch to get one/two (transformation, prob., mag.) tuples
#   Average softmax probability (get softmax of the outputs, then average them to get the probability)


# For every batch, store all outputs. Pick top operations
# Every image -> output 2 operation tuples e.g. 14 trans + 1 prob + 1 mag. 32 output total. 
#   14 neuron output is then prob. of transformations (softmax + average across dim = 0)
#   1000x28 
#   Problem 1: have 28, if we pick argmax top 2

    # For each image have 28 dim output. Calculate covariance of 1000x28 using np.cov(28_dim_vector.T)
    # Give 28x28 covariance matrix. Pick top k pairs (corresponds to largest covariance pairs)
    #   Once we have pairs, go back to 1000x32 output. Find cases where the largest cov. pairs are used and use those probs and mags


# Covariance matrix -> prob. of occurance (might be bad pairs)
# Pair criteria -> highest softmax prob and probaility of occurence

    def get_full_policy(self, x):
        """
        Generates the full policy (5 x 2 subpolicies)
        """
        section = self.auto_aug_agent.fun_num + self.auto_aug_agent.p_bins + self.auto_aug_agent.m_bins
        y = self.auto_aug_agent.forward(x)
        full_policy = []
        for pol in range(self.sub_num_pol):
            int_pol = []
            for _ in range(2):
                idx_ret = torch.argmax(y[:, (pol * section):(pol*section) + self.fun_num].mean(dim = 0))

                trans, need_mag = self.augmentation_space[idx_ret]

                p_ret = (1/(self.p_bins-1)) * torch.argmax(y[:, (pol * section)+self.fun_num:(pol*section)+self.fun_num+self.p_bins].mean(dim = 0))
                mag = torch.argmax(y[:, (pol * section)+self.fun_num+self.p_bins:((pol+1)*section)].mean(dim = 0)) if need_mag else None
                int_pol.append((trans, p_ret, mag))

            full_policy.append(tuple(int_pol))

        return full_policy

    
    def get_policy_cov(self, x):
        """
        Need p_bins = 1, num_sub_pol = 1, mag_bins = 1
        """
        section = self.auto_aug_agent.fun_num + self.auto_aug_agent.p_bins + self.auto_aug_agent.m_bins

        y = self.auto_aug_agent.forward(x) # 1000 x 32

        y_1 = torch.softmax(y[:,:self.auto_aug_agent.fun_num], dim = 1) # 1000 x 14
        y_2 = torch.softmax(y[:,section:section+self.auto_aug_agent.fun_num], dim = 1)
        concat = torch.cat((y_1, y_2), dim = 1)

        cov_mat = torch.cov(concat.T)#[:self.auto_aug_agent.fun_num, self.auto_aug_agent.fun_num:]
        cov_mat = cov_mat[:self.auto_aug_agent.fun_num, self.auto_aug_agent.fun_num:]
        shape_store = cov_mat.shape

        cov_mat = torch.reshape(cov_mat, (1, -1)).squeeze()
        max_idx = torch.argmax(cov_mat)
        val = (max_idx//shape_store[0])
        max_idx = (val, max_idx - (val * shape_store[0]))

        counter, prob1, prob2, mag1, mag2 = (0, 0, 0, 0, 0)

        if self.augmentation_space[max_idx[0]]:
            mag1 = None
        if self.augmentation_space[max_idx[1]]:
            mag2 = None

        for idx in range(y.shape[0]):
            # print("torch.argmax(y_1[idx]): ", torch.argmax(y_1[idx]))
            # print("torch.argmax(y_2[idx]): ", torch.argmax(y_2[idx]))
            # print("max idx0: ", max_idx[0])
            # print("max idx1: ", max_idx[1])

            if (torch.argmax(y_1[idx]) == max_idx[0]) and (torch.argmax(y_2[idx]) == max_idx[1]):
                prob1 += y[idx, self.auto_aug_agent.fun_num+1]
                prob2 += y[idx, section+self.auto_aug_agent.fun_num+1]
                if mag1 is not None:
                    mag1 += y[idx, self.auto_aug_agent.fun_num+2]
                if mag2 is not None:
                    mag2 += y[idx, section+self.auto_aug_agent.fun_num+2]
                counter += 1
        
        prob1 = prob1/counter if counter != 0 else 0
        prob2 = prob2/counter if counter != 0 else 0
        if mag1 is not None:
            mag1 = mag1/counter 
        if mag2 is not None:
            mag2 = mag2/counter            
        
        return [(self.augmentation_space[max_idx[0]], prob1, mag1), (self.augmentation_space[max_idx[1]], prob2, mag2)]


        



    def run_instance(self, return_weights = False):
        """
        Runs the GA instance and returns the model weights as a dictionary
        """
        self.ga_instance.run()
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        if return_weights:
            return torchga.model_weights_as_dict(model=self.auto_aug_agent, weights_vector=solution)
        else:
            return solution, solution_fitness, solution_idx


    def new_model(self):
        """
        Simple function to create a copy of the secondary model (used for classification)
        """
        copy_model = copy.deepcopy(self.child_network)
        return copy_model


    def set_up_instance(self):

        def fitness_func(solution, sol_idx):
            """
            Defines fitness function (accuracy of the model)
            """

            model_weights_dict = torchga.model_weights_as_dict(model=self.auto_aug_agent,
                                                            weights_vector=solution)

            self.auto_aug_agent.load_state_dict(model_weights_dict)

            for idx, (test_x, label_x) in enumerate(train_loader):
                # full_policy = self.get_full_policy(test_x)
                full_policy = self.get_policy_cov(test_x)

            print("full_policy: ", full_policy)
            cop_mod = self.new_model()

            fit_val = test_autoaugment_policy(full_policy, self.train_dataset, self.test_dataset)[0]
            cop_mod = 0

            return fit_val

        def on_generation(ga_instance):
            """
            Just prints stuff while running
            """
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
            return


        self.ga_instance = pygad.GA(num_generations=self.num_generations, 
            num_parents_mating=self.num_parents_mating, 
            initial_population=self.initial_population,
            fitness_func=fitness_func,
            on_generation = on_generation)







auto_aug_agent = Learner()
ev_learner = Evolutionary_learner(auto_aug_agent, train_loader=train_loader, child_network=LeNet(), augmentation_space=augmentation_space, p_bins=1, mag_bins=1, sub_num_pol=1)
ev_learner.run_instance()


solution, solution_fitness, solution_idx = ev_learner.ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
# Fetch the parameters of the best solution.
best_solution_weights = torchga.model_weights_as_dict(model=ev_learner.auto_aug_agent,
                                                      weights_vector=solution)