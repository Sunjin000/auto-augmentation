# The parent class for all other autoaugment learners

import torch
import torch.nn as nn
import torch.optim as optim
from MetaAugment.main import train_child_network, create_toy
from MetaAugment.autoaugment_learners.autoaugment import AutoAugment

import torchvision.transforms as transforms

from pprint import pprint
import matplotlib.pyplot as plt


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

        self.op_tensor_length = fun_num+p_bins+m_bins if discrete_p_m else fun_num+2

        # should we repre
        self.discrete_p_m = discrete_p_m

        # TODO: We should probably use a different way to store results than self.history
        self.history = []


    def translate_operation_tensor(self, operation_tensor, return_log_prob=False, argmax=False):
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

            return_log_prob (boolesn): 
                                When this is on, we return which indices (of fun, prob, mag) were
                                chosen (either randomly or deterministically, depending on argmax).
                                This is used, for example, in the gru_learner to calculate the
                                probability of the actions were chosen, which is then logged, then
                                differentiated.

            argmax (boolean): 
                            Whether we are taking the argmax of the softmaxed tensors. 
                            If this is False, we treat the softmaxed outputs as multinomial pdf's.

        Returns:
            operation (list of tuples):
                                An operation in the format that can be directly put into an
                                AutoAugment object.
            log_prob
                                
        '''
        if (not self.discrete_p_m) and return_log_prob:
            raise ValueError("You are not supposed to use return_log_prob=True when the agent's \
                            self.discrete_p_m is False!")

        # make sure shape is correct
        assert operation_tensor.shape==(self.op_tensor_length, ), operation_tensor.shape

        # if probability and magnitude are represented as discrete variables
        if self.discrete_p_m:
            fun_t, prob_t, mag_t = operation_tensor.split([self.fun_num, self.p_bins, self.m_bins])

            # make sure they are of right size
            assert fun_t.shape==(self.fun_num,), f'{fun_t.shape} != {self.fun_num}'
            assert prob_t.shape==(self.p_bins,), f'{prob_t.shape} != {self.p_bins}'
            assert mag_t.shape==(self.m_bins,), f'{mag_t.shape} != {self.m_bins}'


            if argmax==True:
                fun_idx = torch.argmax(fun_t).item()
                prob_idx = torch.argmax(prob_t).item() # 0 <= p <= 10
                mag = torch.argmax(mag_t).item() # 0 <= m <= 9
            elif argmax==False:
                # we need these to add up to 1 to be valid pdf's of multinomials
                assert torch.sum(fun_t).isclose(torch.ones(1)), torch.sum(fun_t)
                assert torch.sum(prob_t).isclose(torch.ones(1)), torch.sum(prob_t)
                assert torch.sum(mag_t).isclose(torch.ones(1)), torch.sum(mag_t)

                fun_idx = torch.multinomial(fun_t, 1).item() # 0 <= fun <= self.fun_num-1
                prob_idx = torch.multinomial(prob_t, 1).item() # 0 <= p <= 10
                mag = torch.multinomial(mag_t, 1).item() # 0 <= m <= 9

            function = augmentation_space[fun_idx][0]
            prob = prob_idx/10

            indices = (fun_idx, prob_idx, mag)

            # log probability is the sum of the log of the softmax values of the indices 
            # (of fun_t, prob_t, mag_t) that we have chosen
            log_prob = torch.log(fun_t[fun_idx]) + torch.log(prob_t[prob_idx]) + torch.log(mag_t[mag])


        # if probability and magnitude are represented as continuous variables
        else:
            fun_t, prob, mag = operation_tensor.split([self.fun_num, 1, 1])
            prob = prob.item()
            # 0 =< prob =< 1
            mag = mag.item()
            # 0 =< mag =< 9

            # make sure the shape is correct
            assert fun_t.shape==(self.fun_num,), f'{fun_t.shape} != {self.fun_num}'
            
            if argmax==True:
                fun_idx = torch.argmax(fun_t)
            elif argmax==False:
                assert torch.sum(fun_t).isclose(torch.ones(1))
                fun_idx = torch.multinomial(fun_t, 1).item()
            prob = round(prob, 1) # round to nearest first decimal digit
            mag = round(mag) # round to nearest integer
            
        function = augmentation_space[fun_idx][0]

        assert 0 <= prob <= 1
        assert 0 <= mag <= self.m_bins-1
        
        # if the image function does not require a magnitude, we set the magnitude to None
        if augmentation_space[fun_idx][1] == True: # if the image function has a magnitude
            operation = (function, prob, mag)
        else:
            operation =  (function, prob, None)
        
        if return_log_prob:
            return operation, log_prob
        else:
            return operation
        

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
    

    def test_autoaugment_policy(self, policy, child_network, train_dataset, test_dataset, 
                                toy_flag, logging=False):
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
                                                batch_size=64,
                                                n_samples=0.04,
                                                seed=100)

        # train the child network with the dataloaders equipped with our specific policy
        accuracy = train_child_network(child_network, 
                                    train_loader, 
                                    test_loader, 
                                    sgd = optim.SGD(child_network.parameters(), lr=3e-1),
                                    # sgd = optim.Adadelta(child_network.parameters(), lr=1e-2),
                                    cost = nn.CrossEntropyLoss(),
                                    max_epochs = 3000000, 
                                    early_stop_num = 60, 
                                    logging = logging)
        
        # if logging is true, 'accuracy' is actually a tuple: (accuracy, accuracy_log)
        return accuracy