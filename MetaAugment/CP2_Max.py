from cgi import test
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
import math

import math
import torch

from enum import Enum
from torch import Tensor
from typing import List, Tuple, Optional, Dict

from torchvision.transforms import functional as F, InterpolationMode

# import MetaAugment.child_networks as child_networks
# from main import *
# from autoaugment_learners.autoaugment import *


# np.random.seed(0)
# random.seed(0)


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


# class LeNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(256, 120)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(120, 84)
#         self.relu4 = nn.ReLU()
#         self.fc3 = nn.Linear(84, 10)
#         self.relu5 = nn.ReLU()

#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.relu1(y)
#         y = self.pool1(y)
#         y = self.conv2(y)
#         y = self.relu2(y)
#         y = self.pool2(y)
#         y = y.view(y.shape[0], -1)
#         y = self.fc1(y)
#         y = self.relu3(y)
#         y = self.fc2(y)
#         y = self.relu4(y)
#         y = self.fc3(y)
#         return y


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 10)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.reshape((-1, 784))
        y = self.fc1(x)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        return y



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

    def __init__(self, network, num_solutions = 10, num_generations = 5, num_parents_mating = 5, train_loader = None, child_network = None, p_bins = 11, mag_bins = 10, sub_num_pol=5, fun_num = 14, augmentation_space = None, train_dataset = None, test_dataset = None):
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


    def get_full_policy(self, x):
        """
        Generates the full policy (self.num_sub_pol subpolicies). Network architecture requires
        output size 5 * 2 * (self.fun_num + self.p_bins + self.mag_bins)

        Parameters 
        -----------
        x -> PyTorch tensor
            Input data for network 

        Returns
        ----------
        full_policy -> [((String, float, float), (String, float, float)), ...)
            Full policy consisting of tuples of subpolicies. Each subpolicy consisting of
            two transformations, with a probability and magnitude float for each
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

    
    def get_policy_cov(self, x, alpha = 0.5):
        """
        Selects policy using population and covariance matrices. For this method 
        we require p_bins = 1, num_sub_pol = 1, mag_bins = 1. 

        Parameters
        ------------
        x -> PyTorch Tensor
            Input data for the AutoAugment network 

        alpha -> Float
            Proportion for covariance and population matrices 

        Returns
        -----------
        Subpolicy -> [(String, float, float), (String, float, float)]
            Subpolicy consisting of two tuples of policies, each with a string associated 
            to a transformation, a float for a probability, and a float for a magnittude
        """
        section = self.auto_aug_agent.fun_num + self.auto_aug_agent.p_bins + self.auto_aug_agent.m_bins

        y = self.auto_aug_agent.forward(x) # 1000 x 32

        y_1 = torch.softmax(y[:,:self.auto_aug_agent.fun_num], dim = 1) # 1000 x 14
        y[:,:self.auto_aug_agent.fun_num] = y_1
        y_2 = torch.softmax(y[:,section:section+self.auto_aug_agent.fun_num], dim = 1)
        y[:,section:section+self.auto_aug_agent.fun_num] = y_2
        concat = torch.cat((y_1, y_2), dim = 1)

        cov_mat = torch.cov(concat.T)#[:self.auto_aug_agent.fun_num, self.auto_aug_agent.fun_num:]
        cov_mat = cov_mat[:self.auto_aug_agent.fun_num, self.auto_aug_agent.fun_num:]
        shape_store = cov_mat.shape

        counter, prob1, prob2, mag1, mag2 = (0, 0, 0, 0, 0)


        prob_mat = torch.zeros(shape_store)
        for idx in range(y.shape[0]):
            prob_mat[torch.argmax(y_1[idx])][torch.argmax(y_2[idx])] += 1
        prob_mat = prob_mat / torch.sum(prob_mat)

        cov_mat = (alpha * cov_mat) + ((1 - alpha)*prob_mat)

        cov_mat = torch.reshape(cov_mat, (1, -1)).squeeze()
        max_idx = torch.argmax(cov_mat)
        val = (max_idx//shape_store[0])
        max_idx = (val, max_idx - (val * shape_store[0]))


        if not self.augmentation_space[max_idx[0]][1]:
            mag1 = None
        if not self.augmentation_space[max_idx[1]][1]:
            mag2 = None
    
        for idx in range(y.shape[0]):
            if (torch.argmax(y_1[idx]) == max_idx[0]) and (torch.argmax(y_2[idx]) == max_idx[1]):
                prob1 += torch.sigmoid(y[idx, self.auto_aug_agent.fun_num]).item()
                prob2 += torch.sigmoid(y[idx, section+self.auto_aug_agent.fun_num]).item()
                if mag1 is not None:
                    mag1 += min(max(0, (y[idx, self.auto_aug_agent.fun_num+1]).item()), 8)
                if mag2 is not None:
                    mag2 += min(max(0, y[idx, section+self.auto_aug_agent.fun_num+1].item()), 8)
                counter += 1

        prob1 = prob1/counter if counter != 0 else 0
        prob2 = prob2/counter if counter != 0 else 0
        if mag1 is not None:
            mag1 = mag1/counter 
        if mag2 is not None:
            mag2 = mag2/counter    

        
        return [(self.augmentation_space[max_idx[0]][0], prob1, mag1), (self.augmentation_space[max_idx[1]][0], prob2, mag2)]


        



    def run_instance(self, return_weights = False):
        """
        Runs the GA instance and returns the model weights as a dictionary

        Parameters
        ------------
        return_weights -> Bool
            Determines if the weight of the GA network should be returned 
        
        Returns
        ------------
        If return_weights:
            Network weights -> Dictionary
        
        Else:
            Solution -> Best GA instance solution

            Solution fitness -> Float

            Solution_idx -> Int
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
        """
        Initialises GA instance, as well as fitness and on_generation functions
        
        """

        def fitness_func(solution, sol_idx):
            """
            Defines the fitness function for the parent selection

            Parameters
            --------------
            solution -> GA solution instance (parsed automatically)

            sol_idx -> GA solution index (parsed automatically)

            Returns 
            --------------
            fit_val -> float            
            """

            model_weights_dict = torchga.model_weights_as_dict(model=self.auto_aug_agent,
                                                            weights_vector=solution)

            self.auto_aug_agent.load_state_dict(model_weights_dict)

            for idx, (test_x, label_x) in enumerate(train_loader):
                full_policy = self.get_policy_cov(test_x)

            fit_val = ((test_autoaugment_policy(full_policy, self.train_dataset, self.test_dataset)[0])/
                        + test_autoaugment_policy(full_policy, self.train_dataset, self.test_dataset)[0]) / 2

            return fit_val

        def on_generation(ga_instance):
            """
            Prints information of generational fitness

            Parameters 
            -------------
            ga_instance -> GA instance

            Returns
            -------------
            None
            """
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
            return


        self.ga_instance = pygad.GA(num_generations=self.num_generations, 
            num_parents_mating=self.num_parents_mating, 
            initial_population=self.initial_population,
            mutation_percent_genes = 0.1,
            fitness_func=fitness_func,
            on_generation = on_generation)




# HEREHEREHERE0

def create_toy(train_dataset, test_dataset, batch_size, n_samples, seed=100):
    # shuffle and take first n_samples %age of training dataset
    shuffle_order_train = np.random.RandomState(seed=seed).permutation(len(train_dataset))
    shuffled_train_dataset = torch.utils.data.Subset(train_dataset, shuffle_order_train)
    
    indices_train = torch.arange(int(n_samples*len(train_dataset)))
    reduced_train_dataset = torch.utils.data.Subset(shuffled_train_dataset, indices_train)
    
    # shuffle and take first n_samples %age of test dataset
    shuffle_order_test = np.random.RandomState(seed=seed).permutation(len(test_dataset))
    shuffled_test_dataset = torch.utils.data.Subset(test_dataset, shuffle_order_test)

    big = 4 # how much bigger is the test set

    indices_test = torch.arange(int(n_samples*len(test_dataset)*big))
    reduced_test_dataset = torch.utils.data.Subset(shuffled_test_dataset, indices_test)

    # push into DataLoader
    train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(reduced_test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def train_child_network(child_network, train_loader, test_loader, sgd,
                            cost, max_epochs=2000, early_stop_num = 5, logging=False,
                            print_every_epoch=True):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    child_network = child_network.to(device=device)
    
    best_acc=0
    early_stop_cnt = 0
    
    # logging accuracy for plotting
    acc_log = [] 

    # train child_network and check validation accuracy each epoch
    for _epoch in range(max_epochs):

        # train child_network
        child_network.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            # onto device
            train_x = train_x.to(device=device, dtype=train_x.dtype)
            train_label = train_label.to(device=device, dtype=train_label.dtype)

            # label_np = np.zeros((train_label.shape[0], 10))

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
                # onto device
                test_x = test_x.to(device=device, dtype=test_x.dtype)
                test_label = test_label.to(device=device, dtype=test_label.dtype)

                predict_y = child_network(test_x.float()).detach()
                predict_ys = torch.argmax(predict_y, axis=-1)
    
                _ = predict_ys == test_label
                correct += torch.sum(_, axis=-1)

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
            print('main.train_child_network best accuracy: ', best_acc)
            break
        
        # if print_every_epoch:
            # print('main.train_child_network best accuracy: ', best_acc)
        acc_log.append(acc)

    if logging:
        return best_acc.item(), acc_log
    return best_acc.item()

def test_autoaugment_policy(subpolicies, train_dataset, test_dataset):

    aa_transform = AutoAugment()
    aa_transform.subpolicies = subpolicies

    train_transform = transforms.Compose([
                                            aa_transform,
                                            transforms.ToTensor()
                                        ])

    train_dataset.transform = train_transform

    # create toy dataset from above uploaded data
    train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size=32, n_samples=0.1)

    child_network = LeNet()
    sgd = optim.SGD(child_network.parameters(), lr=1e-1)
    cost = nn.CrossEntropyLoss()

    best_acc, acc_log = train_child_network(child_network, train_loader, test_loader,
                                                sgd, cost, max_epochs=100, logging=True)

    return best_acc, acc_log



__all__ = ["AutoAugmentPolicy", "AutoAugment", "RandAugment", "TrivialAugmentWide"]


def _apply_op(img: Tensor, op_name: str, magnitude: float,
                interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                        interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                        interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                        interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                        interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class AutoAugmentPolicy(Enum):
    """AutoAugment policies learned on different datasets.
    Available policies are IMAGENET, CIFAR10 and SVHN.
    """
    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"


# FIXME: Eliminate copy-pasted code for fill standardization and _augmentation_space() by moving stuff on a base class
class AutoAugment(torch.nn.Module):
    r"""AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        policy (AutoAugmentPolicy): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.subpolicies = self._get_subpolicies(policy)

    def _get_subpolicies(
        self,
        policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError("The provided policy {} is not recognized.".format(policy))

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "Invert": (torch.tensor(0.0), False),
        }

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(torch.randint(transform_num, (1,)).item())
        probs = torch.rand((2,))
        signs = torch.randint(2, (2,))

        return policy_id, probs, signs

    def forward(self, img: Tensor, dis_mag = True) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        transform_id, probs, signs = self.get_params(len(self.subpolicies))

        for i, (op_name, p, magnitude) in enumerate(self.subpolicies):
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)


        return img

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(policy={}, fill={})'.format(self.policy, self.fill)


class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31,
                    interpolation: InterpolationMode = InterpolationMode.NEAREST,
                    fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        for _ in range(self.num_ops):
            op_meta = self._augmentation_space(self.num_magnitude_bins, F.get_image_size(img))
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_ops={num_ops}'
        s += ', magnitude={magnitude}'
        s += ', num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)


class TrivialAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_magnitude_bins: int = 31, interpolation: InterpolationMode = InterpolationMode.NEAREST,
                    fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
            if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)

# HEREHEREHEREHERE1








train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, download=False, 
                            transform=None)
test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, download=False,
                            transform=torchvision.transforms.ToTensor())


auto_aug_agent = Learner()
ev_learner = Evolutionary_learner(auto_aug_agent, train_loader=train_loader, child_network=LeNet(), augmentation_space=augmentation_space, p_bins=1, mag_bins=1, sub_num_pol=1, train_dataset=train_dataset, test_dataset=test_dataset)
ev_learner.run_instance()


solution, solution_fitness, solution_idx = ev_learner.ga_instance.best_solution()

print(f"Best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")
# Fetch the parameters of the best solution.
best_solution_weights = torchga.model_weights_as_dict(model=ev_learner.auto_aug_agent,
                                                        weights_vector=solution)
