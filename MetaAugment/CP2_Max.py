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
    def __init__(self, fun_num=14, p_bins=11, m_bins=10):
        self.fun_num = fun_num
        self.p_bins = p_bins 
        self.m_bins = m_bins 

        self.augmentation_space = [
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
        self.fc3 = nn.Linear(84, 5 * 2 * (self.fun_num + self.p_bins + self.m_bins))

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

    def get_idx(self, x):
        section = self.fun_num + self.p_bins + self.m_bins
        y = self.forward(x)
        full_policy = []
        for pol in range(5 * 2):
            int_pol = []
            idx_ret = torch.argmax(y[:, (pol * section):(pol*section) + self.fun_num].mean(dim = 0))

            trans, need_mag = self.augmentation_space[idx_ret]

            p_ret = 0.1 * torch.argmax(y[:, (pol * section)+self.fun_num:(pol*section)+self.fun_num+self.p_bins].mean(dim = 0))
            mag = torch.argmax(y[:, (pol * section)+self.fun_num+self.p_bins:((pol+1)*section)].mean(dim = 0)) if need_mag else 0
            int_pol.append((trans, p_ret, mag))
            if pol % 2 != 0:
                full_policy.append(tuple(int_pol))

        return full_policy


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
def train_model(full_policy, child_network):
    """
    Takes in the specific transformation index and probability 
    """

    # transformation = generate_policy(5, ps, mags)

    train_transform = transforms.Compose([
                                            full_policy,
                                            transforms.ToTensor()
                                        ])

    batch_size = 32
    n_samples = 0.005

    train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, download=False, transform=train_transform)
    test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, download=False, transform=torchvision.transforms.ToTensor())

    train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size, 0.01)


    sgd = optim.SGD(child_network.parameters(), lr=1e-1)
    cost = nn.CrossEntropyLoss()
    epoch = 20


    best_acc = train_child_network(child_network, train_loader, test_loader,
                                     sgd, cost, max_epochs=100, print_every_epoch=False)

    return best_acc





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

    def __init__(self, network, num_solutions = 30, num_generations = 10, num_parents_mating = 15, train_loader = None, sec_model = None, p_bins = 11, mag_bins = 10, fun_num = 14):
        self.meta_rl_agent = Learner(fun_num, p_bins=11, m_bins=10)
        self.torch_ga = torchga.TorchGA(model=network, num_solutions=num_solutions)
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.initial_population = self.torch_ga.population_weights
        self.train_loader = train_loader
        self.sec_model = sec_model
        self.p_bins = p_bins 
        self.mag_bins = mag_bins
        self.fun_num = fun_num

        assert num_solutions > num_parents_mating, 'Number of solutions must be larger than the number of parents mating!'

        self.set_up_instance()
    

    def generate_policy(self, sp_num, ps, mags):
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


    def run_instance(self, return_weights = False):
        self.ga_instance.run()
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        if return_weights:
            return torchga.model_weights_as_dict(model=self.meta_rl_agent, weights_vector=solution)
        else:
            return solution, solution_fitness, solution_idx


    def new_model(self):
        copy_model = copy.deepcopy(self.sec_model)
        return copy_model


    def set_up_instance(self):

        def fitness_func(solution, sol_idx):
            """
            Defines fitness function (accuracy of the model)
            """
            model_weights_dict = torchga.model_weights_as_dict(model=self.meta_rl_agent,
                                                            weights_vector=solution)
            self.meta_rl_agent.load_state_dict(model_weights_dict)
            for idx, (test_x, label_x) in enumerate(train_loader):
                full_policy = self.meta_rl_agent.get_idx(test_x)
            cop_mod = self.new_model()
            fit_val = train_model(full_policy, cop_mod)
            cop_mod = 0
            return fit_val

        def on_generation(ga_instance):
            """
            Just prints stuff while running
            """
            print("Generation = {generation}".format(generation=self.ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=self.ga_instance.best_solution()[1]))
            return


        self.ga_instance = pygad.GA(num_generations=self.num_generations, 
            num_parents_mating=self.num_parents_mating, 
            initial_population=self.initial_population,
            fitness_func=fitness_func,
            on_generation = on_generation)


meta_rl_agent = Learner()
ev_learner = Evolutionary_learner(meta_rl_agent, train_loader=train_loader, sec_model=LeNet())
ev_learner.run_instance()


solution, solution_fitness, solution_idx = ev_learner.ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
# Fetch the parameters of the best solution.
best_solution_weights = torchga.model_weights_as_dict(model=ev_learner.meta_rl_agent,
                                                      weights_vector=solution)