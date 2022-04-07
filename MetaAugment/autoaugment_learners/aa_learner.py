# The parent class for all other autoaugment learners``

import torch
import numpy as np
from MetaAugment.main import *
import MetaAugment.child_networks as cn
import torchvision.transforms as transforms
from MetaAugment.autoaugment_learners.autoaugment import *

import torchvision.transforms.autoaugment as torchaa
from torchvision.transforms import functional as F, InterpolationMode

from pprint import pprint

# We will use this augmentation_space temporarily. Later on we will need to 
# make sure we are able to add other image functions if the users want.
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


class aa_learner:
    def __init__(self, sp_num=5, fun_num=14, p_bins=11, m_bins=10, discrete_p_m=False):
        '''
        Args:
            spdim (int): number of subpolicies per policy
            fun_num (int): number of image functions in our search space
            p_bins (int): number of bins we divide the interval [0,1] for probabilities
            m_bins (int): number of bins we divide the magnitude space

            discrete_p_m (boolean): Whether or not the agent should represent probability and 
                                    magnitude as discrete variables as the out put of the 
                                    controller (A controller can be a neural network, genetic
                                    algorithm, etc.)
        '''
        self.sp_num = sp_num
        self.fun_num = fun_num
        self.p_bins = p_bins
        self.m_bins = m_bins

        # should we repre
        self.discrete_p_m = discrete_p_m

        # TODO: We should probably use a different way to store results than self.history
        self.history = []


    def translate_operation_tensor(self, operation_tensor, argmax=False):
        '''
        takes in a tensor representing an operation and returns an actual operation which
        is in the form of:
            ("Invert", 0.8, None)
            or
            ("Contrast", 0.2, 6)

        Args:
            operation_tensor (tensor): 
                                We expect this tensor to already have been softmaxed.
                                Furthermore,
                                - If self.discrete_p_m is True, we expect to take in a tensor with
                                dimension (self.fun_num + self.p_bins + self.m_bins)
                                - If self.discrete_p_m is False, we expect to take in a tensor with
                                dimension (self.fun_num + 1 + 1)

            argmax (boolean): 
                            Whether we are taking the argmax of the softmaxed tensors. 
                            If this is False, we treat the softmaxed outputs as multinomial pdf's.
        '''
        # if probability and magnitude are represented as discrete variables
        if self.discrete_p_m:
            fun_t = operation_tensor[ : self.fun_num]
            prob_t = operation_tensor[self.fun_num : self.fun_num+self.p_bins]
            mag_t = operation_tensor[-self.m_bins : ]

            # make sure they are of right size
            assert fun_t.shape==(self.fun_num,), f'{fun_t.shape} != {self.fun_num}'
            assert prob_t.shape==(self.p_bins,), f'{prob_t.shape} != {self.p_bins}'
            assert mag_t.shape==(self.m_bins,), f'{mag_t.shape} != {self.m_bins}'

            if argmax==True:
                fun = torch.argmax(fun_t)
                prob = torch.argmax(prob_t) # 0 <= p <= 10
                mag = torch.argmax(mag_t) # 0 <= m <= 9
            elif argmax==False:
                # we need these to add up to 1 to be valid pdf's of multinomials
                assert torch.sum(fun_t)==1
                assert torch.sum(prob_t)==1
                assert torch.sum(mag_t)==1
                fun = torch.multinomial(fun_t, 1) # 0 <= fun <= self.fun_num-1
                prob = torch.multinomial(prob_t, 1) # 0 <= p <= 10
                mag = torch.multinomial(mag_t, 1) # 0 <= m <= 9

            function = augmentation_space[fun][0]
            prob = prob/10


        # if probability and magnitude are represented as continuous variables
        else:
            fun_t = operation_tensor[:self.fun_num]
            p = operation_tensor[-2].item() # 0 < p < 1
            m = operation_tensor[-1].item() # 0 < m < 9

            # make sure the shape is correct
            assert fun_t.shape==(self.fun_num,), f'{fun_t.shape} != {self.fun_num}'
            
            if argmax==True:
                fun = torch.argmax(fun_t)
            elif argmax==False:
                assert torch.sum(fun_t)==1
                fun = torch.multinomial(fun_t, 1)
            
            function = augmentation_space[fun][0]
            prob = round(p, 1) # round to nearest first decimal digit
            mag = round(m) # round to nearest integer
            # If argmax is False, we treat operation_tensor as a concatenation of three
            # multinomial pdf's.

        assert 0 <= prob <= 1
        assert 0 <= mag <= self.m_bins-1
        
        # if the image function does not require a magnitude, we set the magnitude to None
        if augmentation_space[fun][1] == True: # if the image function has a magnitude
            return (function, prob, mag)
        else:
            return (function, prob, None)
            



    def generate_new_policy(self):
        '''
        Generate a new random policy in the form of
            [
            (("Invert", 0.8, None), ("Contrast", 0.2, 6)),
            (("Rotate", 0.7, 2), ("Invert", 0.8, None)),
            (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
            (("ShearY", 0.5, 8), ("Invert", 0.7, None)),
            ]
        '''
        raise NotImplementedError('generate_new_policy not implemented in aa_learner')


    def learn(self, train_dataset, test_dataset, child_network_architecture, toy_flag):
        '''
        Does the loop which is seen in Figure 1 in the AutoAugment paper.
        In other words, repeat:
            1. <generate a random policy>
            2. <see how good that policy is>
            3. <save how good the policy is in a list/dictionary>
        until a certain condition (either specified by the user or pre-specified) is met
        '''

        # This is dummy code
        # test out 15 random policies
        for _ in range(15):
            policy = self.generate_new_policy()

            pprint(policy)
            child_network = child_network_architecture()
            reward = self.test_autoaugment_policy(policy, child_network, train_dataset,
                                                test_dataset, toy_flag)

            self.history.append((policy, reward))
    

    def test_autoaugment_policy(self, policy, child_network, train_dataset, test_dataset, toy_flag):
        '''
        Given a policy (using AutoAugment paper terminology), we train a child network
        using the policy and return the accuracy (how good the policy is for the dataset and 
        child network).
        '''
        # We need to define an object aa_transform which takes in the image and 
        # transforms it with the policy (specified in its .policies attribute)
        # in its forward pass
        aa_transform = AutoAugment()
        aa_transform.subpolicies = policy
        train_transform = transforms.Compose([
                                                aa_transform,
                                                transforms.ToTensor()
                                            ])
        
        # We feed the transformation into the Dataset object
        train_dataset.transform = train_transform

        # create Dataloader objects out of the Dataset objects
        train_loader, test_loader = create_toy(train_dataset,
                                                test_dataset,
                                                batch_size=32,
                                                n_samples=0.01,
                                                seed=100)

        # train the child network with the dataloaders equipped with our specific policy
        accuracy = train_child_network(child_network, 
                                    train_loader, 
                                    test_loader, 
                                    sgd = optim.SGD(child_network.parameters(), lr=1e-1),
                                    cost = nn.CrossEntropyLoss(),
                                    max_epochs = 100, 
                                    early_stop_num = 15, 
                                    logging = False)
        return accuracy