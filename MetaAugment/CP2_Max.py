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

# import MetaAugment.child_networks as child_networks
# from MetaAugment.main import *


np.random.seed(0)
random.seed(0)


class Learner(nn.Module):
    def __init__(self, num_transforms = 3):
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
        self.fc3 = nn.Linear(84, 13)
        # self.sig = nn.Sigmoid()

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
        y = self.forward(x)
        idx_ret = torch.argmax(y[:, 0:3].mean(dim = 0))
        p_ret = 0.1 * torch.argmax(y[:, 3:].mean(dim = 0))
        return (idx_ret, p_ret)

        # return (torch.argmax(y[0:3]), y[torch.argmax(y[3:])])

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
def train_model(transform_idx, p, child_network):
    """
    Takes in the specific transformation index and probability 
    """

    if transform_idx == 0:
        transform_train = torchvision.transforms.Compose(
           [
            torchvision.transforms.RandomVerticalFlip(p),
            torchvision.transforms.ToTensor(),
            ]
               )
    elif transform_idx == 1:
        transform_train = torchvision.transforms.Compose(
           [
            torchvision.transforms.RandomHorizontalFlip(p),
            torchvision.transforms.ToTensor(),
            ]
               )
    else:
        transform_train = torchvision.transforms.Compose(
           [
            torchvision.transforms.RandomGrayscale(p),
            torchvision.transforms.ToTensor(),
            ]
               )

    batch_size = 32
    n_samples = 0.005

    train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, download=False, transform=transform_train)
    test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, download=False, transform=torchvision.transforms.ToTensor())

    train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size, 0.01)


    # child_network = child_networks.lenet()

    sgd = optim.SGD(child_network.parameters(), lr=1e-1)
    cost = nn.CrossEntropyLoss()
    epoch = 20


    best_acc = train_child_network(child_network, train_loader, test_loader, sgd, cost, max_epochs=100)

    return best_acc






def train_child_network(child_network, train_loader, test_loader, sgd,
                         cost, max_epochs=100, early_stop_num = 10, logging=False):
    best_acc=0
    early_stop_cnt = 0
    
    # logging accuracy for plotting
    acc_log = [] 

    # train child_network and check validation accuracy each epoch
    for _epoch in range(max_epochs):

        # train child_network
        child_network.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = child_network(train_x.float())
            loss = cost(predict_y, train_label.long())
            loss.backward()
            sgd.step()

        # check validation accuracy on validation set
        correct = 0
        _sum = 0
        child_network.eval()
        with torch.no_grad():
            for idx, (test_x, test_label) in enumerate(test_loader):
                predict_y = child_network(test_x.float()).detach()
                predict_ys = np.argmax(predict_y, axis=-1)
                label_np = test_label.numpy()
                _ = predict_ys == test_label
                correct += np.sum(_.numpy(), axis=-1)
                _sum += _.shape[0]
        
        # update best validation accuracy if it was higher, otherwise increase early stop count
        acc = correct / _sum


        if acc > best_acc :
            best_acc = acc
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        # exit if validation gets worse over 10 runs
        if early_stop_cnt >= early_stop_num:
            break
        
        # print('main.train_child_network best accuracy: ', best_acc)
        acc_log.append(acc)

    if logging:
        return best_acc, acc_log
    return best_acc

def create_toy(train_dataset, test_dataset, batch_size, n_samples, seed=100):
    # shuffle and take first n_samples %age of training dataset
    shuffle_order_train = np.random.RandomState(seed=seed).permutation(len(train_dataset))
    shuffled_train_dataset = torch.utils.data.Subset(train_dataset, shuffle_order_train)
    
    indices_train = torch.arange(int(n_samples*len(train_dataset)))
    reduced_train_dataset = torch.utils.data.Subset(shuffled_train_dataset, indices_train)
    
    # shuffle and take first n_samples %age of test dataset
    shuffle_order_test = np.random.RandomState(seed=seed).permutation(len(test_dataset))
    shuffled_test_dataset = torch.utils.data.Subset(test_dataset, shuffle_order_test)

    indices_test = torch.arange(int(n_samples*len(test_dataset)))
    reduced_test_dataset = torch.utils.data.Subset(shuffled_test_dataset, indices_test)

    # push into DataLoader
    train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(reduced_test_dataset, batch_size=batch_size)

    return train_loader, test_loader


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

    def __init__(self, network, num_solutions = 30, num_generations = 10, num_parents_mating = 15, train_loader = None, sec_model = None):
        self.meta_rl_agent = network
        self.torch_ga = torchga.TorchGA(model=network, num_solutions=num_solutions)
        self.num_generations = num_generations
        self.num_parents_mating = num_parents_mating
        self.initial_population = self.torch_ga.population_weights
        self.train_loader = train_loader
        self.backup_model = sec_model

        assert num_solutions > num_parents_mating, 'Number of solutions must be larger than the number of parents mating!'

        self.set_up_instance()
    

    def run_instance(self, return_weights = False):
        self.ga_instance.run()
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        if return_weights:
            return torchga.model_weights_as_dict(model=self.meta_rl_agent, weights_vector=solution)
        else:
            return solution, solution_fitness, solution_idx

    def new_model(self):
        copy_model = copy.deepcopy(self.backup_model)
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
                trans_idx, p = self.meta_rl_agent.get_idx(test_x)
            cop_mod = self.new_model()
            fit_val = train_model(trans_idx, p, cop_mod)
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