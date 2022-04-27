import torch
import numpy as np

import MetaAugment.child_networks as cn
from MetaAugment.autoaugment_learners.aa_learner import aa_learner
import random


class Genetic_learner(aa_learner):

    def __init__(self, 
                # search space settings
                sp_num=5,
                p_bins=11, 
                m_bins=10, 
                discrete_p_m=False,
                exclude_method=[],
                # child network settings
                learning_rate=1e-1, 
                max_epochs=float('inf'),
                early_stop_num=20,
                batch_size=8,
                toy_size=1,
                num_offspring=1, 
                ):

        super().__init__(
                    sp_num=sp_num, 
                    p_bins=p_bins, 
                    m_bins=m_bins, 
                    discrete_p_m=discrete_p_m, 
                    batch_size=batch_size, 
                    toy_size=toy_size, 
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    early_stop_num=early_stop_num,
                    exclude_method=exclude_method
                    )

        self.bin_to_aug =  {}
        for idx, augmentation in enumerate(self.augmentation_space):
            bin_rep = '{0:b}'.format(idx)
            while len(bin_rep) < len('{0:b}'.format(len(self.augmentation_space))):
                bin_rep = '0' + bin_rep
            self.bin_to_aug[bin_rep] = augmentation[0]
    
        self.mag_to_bin = {
            '0': "0000",
            '1': '0001',
            '2': '0010',
            '3': '0011',
            '4': '0100',
            '5': '0101',
            '6': '0110',
            '7': '0111',
            '8' : '1000',
            '9': '1001',
            '10': '1010',
        }

        self.prob_to_bin = {
            '0': "0000",
            '0.0' : '0000',
            '0.1': '0001',
            '0.2': '0010',
            '0.3': '0011',
            '0.4': '0100',
            '0.5': '0101',
            '0.6': '0110',
            '0.7': '0111',
            '0.8' : '1000',
            '0.9': '1001',
            '1.0': '1010',
            '1' : '1010',
        }

        self.bin_to_prob = dict((value, key) for key, value in self.prob_to_bin.items())
        self.bin_to_mag = dict((value, key) for key, value in self.mag_to_bin.items())
        self.aug_to_bin = dict((value, key) for key, value in self.bin_to_aug.items())

        self.num_offspring = num_offspring


    def gen_random_subpol(self):
        choose_items = [x[0] for x in self.augmentation_space]
        trans1 = str(random.choice(choose_items))
        trans2 = str(random.choice(choose_items))
        prob1 = float(random.randrange(0, 11, 1) / 10)
        prob2 = float(random.randrange(0, 11, 1) / 10)
        mag1 = int(random.randrange(0, 10, 1))
        mag2 = int(random.randrange(0, 10, 1))
        subpol = ((trans1, prob1, mag1), (trans2, prob2, mag2))
        return subpol


    def gen_random_policy(self):
        pol = []
        for idx in range(self.sp_num):
            pol.append(self.gen_random_subpol())
        return pol

    
    def bin_to_subpol(self, subpol):
        pol = []
        for idx in range(2):
            pol.append((self.bin_to_aug[subpol[idx*12:(idx*12)+4]], float(self.bin_to_prob[subpol[(idx*12)+4: (idx*12)+8]])\
                        , int(self.bin_to_mag[subpol[(idx*12)+8:(idx*12)+12]])))
        pol = tuple(pol)
        return pol   


    def subpol_to_bin(self, subpol):
        pol = ''  
        trans1, prob1, mag1 = subpol[0]
        trans2, prob2, mag2 = subpol[1]
        pol += self.aug_to_bin[trans1] + self.prob_to_bin[prob1] + self.mag_to_bin[mag1] + \
               self.aug_to_bin[trans2] + self.prob_to_bin[prob2] + self.mag_to_bin[mag2]
        return pol


    def choose_parents(self, parents, parents_weights):
        parent1 = random.choices(parents, parents_weights, k=1)
        parent2 = random.choices(parents, parents_weights, k=1)
        while parent2 == parent1:
            parent2 = random.choices(parents, parents_weights, k=1)
        return (parent1, parent2)

    
    def generate_children(self):
        parent_acc = sorted(self.history, key = lambda x: x[1], reverse=True)[:self.sp_num]
        parents = [x[0] for x in parent_acc]
        parents_weights = [x[1] for x in parent_acc]
        new_pols = []
        for _ in range(self.num_offspring):
            parent1, parent2 = self.choose_parents(parents, parents_weights)
            cross_over = random.randint(1, len(parent2), 1)
            parent1[cross_over:] = parent2[cross_over:]
            new_pols.append(self.bin_to_subpol(parent1))
        return new_pols

    
    def learn(self, train_dataset, test_dataset, child_network_architecture, iterations = 10):

        for idx in range(iterations):
            if len(self.history) < self.sp_num:
                policy = [self.gen_random_subpol()]
            else:
                policy = self.generate_children()
            print("policy: ", policy)

            reward = self.test_autoaugment_policy(policy,
                                                child_network_architecture,
                                                train_dataset,
                                                test_dataset)            

            self.history.append((policy, reward))



if __name__=='__main__':
    # We can initialize the train_dataset with its transform as None.
    # Later on, we will change this object's transform attribute to the policy
    # that we want to test
    import torchvision.datasets as datasets
    import torchvision
    
    # train_dataset = datasets.MNIST(root='./datasets/mnist/train',
    #                                 train=True, download=True, transform=None)
    # test_dataset = datasets.MNIST(root='./datasets/mnist/test', 
    #                         train=False, download=True, transform=torchvision.transforms.ToTensor())
    train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=None)
    test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                            train=False, download=True,
                            transform=torchvision.transforms.ToTensor())
    child_network_architecture = cn.lenet
    # child_network_architecture = cn.lenet()

    agent = Genetic_learner(
                                sp_num=1,
                                toy_size=0.01,
                                batch_size=4,
                                learning_rate=0.05,
                                max_epochs=float('inf'),
                                early_stop_num=20,
                                )
    agent.learn(train_dataset,
                test_dataset,
                child_network_architecture=child_network_architecture,
                iterations=3)

    # with open('randomsearch_logs.pkl', 'wb') as file:
    #                 pickle.dump(self.history, file)
    print(agent.history)



