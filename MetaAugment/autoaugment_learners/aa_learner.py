# DUMMY PSEUDOCODE!
# this might become the superclass for all other autoaugment_learners

from importlib import machinery
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
num_bins = 10
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


# TODO: Right now the aa_learner is identical to randomsearch_learner. Change
# this so that it can act as a superclass to all other augment learners
class aa_learner:
    def __init__(self, sp_num=5):
        '''
        Args:
            spdim: number of subpolicies per policy
        '''
        self.sp_num = sp_num

        # fun_num is the number of different operations
        # TODO: Allow fun_num to be changed with the user's specifications 
        self.fun_num = 14

        # TODO: We should probably use a different way to store results than self.history
        self.history = []

    def generate_new_discrete_operation(self, fun_num=14, p_bins=10, m_bins=10):
        '''
        generate a new random operation in the form of a tensor of dimension:
            (fun_num + 11 + 10)

        The first fun_num dimensions is a 1-hot encoding to specify which function to use.
        The next 11 dimensions specify which 'probability' to choose.
            (0.0, 0.1, ..., 1.0)
        The next 10 dimensions specify which 'magnitude' to choose.
            (0, 1, ..., 9)
        '''
        fun = np.random.randint(0, fun_num)
        prob = np.random.randint(p_bins+1, fun_num)
        mag = np.random.randint(m_bins, fun_num)
        
        fun_t= torch.zeros(fun_num)
        fun_t[fun] = 1
        prob_t = torch.zeros(p_bins+1)
        prob_t[prob] = 1
        mag_t = torch.zeros(m_bins)
        mag_t[mag] = 1

        return torch.cat([fun_t, prob_t, mag_t])


    def generate_new_continuous_operation(self, fun_num=14, p_bins=10, m_bins=10):
        '''
        Returns operation_tensor, which is a tensor representation of a random operation with
        dimension:
            (fun_num + 1 + 1)

        The first fun_num dimensions is a 1-hot encoding to specify which function to use.
        The next 1 dimensions specify which 'probability' to choose.
            0 < x < 1
        The next 1 dimensions specify which 'magnitude' to choose.
            0 < x < 9
        '''
        fun = np.random.randint(0, fun_num)
        
        fun_p_m = torch.zeros(fun_num + 2)
        fun_p_m[fun] = 1
        fun_p_m[-2] = np.random.uniform() # 0<prob<1
        fun_p_m[-1] = np.random.uniform() * (m_bins-1) # 0<mag<9
        
        return fun_p_m


    def translate_operation_tensor(self, operation_tensor, fun_num=14,
                                        p_bins=10, m_bins=10,
                                        discrete_p_m=False):
        '''
        takes in a tensor representing a operation and returns an actual operation which
        is in the form of:
            ("Invert", 0.8, None)
            or
            ("Contrast", 0.2, 6)

        Args:
            operation_tensor
            continuous_p_m (boolean): whether the operation_tensor has continuous representations
                                    of probability and magnitude
        '''
        # if input operation_tensor is discrete
        if discrete_p_m:
            fun_t = operation_tensor[:fun_num]
            prob_t = operation_tensor[fun_num:fun_num+p_bins+1]
            mag_t = operation_tensor[-m_bins:]

            fun = torch.argmax(fun_t)
            prob = torch.argmax(prob_t) # 0 <= p <= 10
            mag = torch.argmax(mag_t) # 0 <= m <= 9

            fun = augmentation_space[fun][0]
            prob = prob/10

            return (fun, prob, mag)

        
        # process continuous operation_tensor
        fun_t = operation_tensor[:fun_num]
        p = operation_tensor[-2].item() # 0 < p < 1
        m = operation_tensor[-1].item() # 0 < m < 9

        fun_num = torch.argmax(fun_t)
        function = augmentation_space[fun_num][0]
        p = round(p, 1) # round to nearest first decimal digit
        m = round(m) # round to nearest integer
        return (function, p, m)


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

        new_policy = []
        for _ in range(self.sp_num):
            # generate 2 operations for each subpolicy
            ops = []
            for i in range(2):
                new_op = self.generate_new_continuous_operation(self.fun_num)
                new_op = self.translate_operation_tensor(new_op)
                ops.append(new_op)

            new_subpolicy = tuple(ops)

            new_policy.append(new_subpolicy)

        return new_policy


    def learn(self, train_dataset, test_dataset, child_network_architecture, toy_flag):
        '''
        Does the loop which is seen in Figure 1 in the AutoAugment paper.
        In other words, repeat:
            1. <generate a random policy>
            2. <see how good that policy is>
            3. <save how good the policy is in a list/dictionary>
        '''

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
        with the policy and return the accuracy.
        '''
        # We need to define an object aa_transform which takes in the image and 
        # transforms it with the policy (specified in its .policies attribute)
        # in its forward pass
        aa_transform = AutoAugment()
        aa_transform.policies = policy
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