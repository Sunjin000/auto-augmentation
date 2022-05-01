import torch
import numpy as np

import autoaug.child_networks as cn
from autoaug.autoaugment_learners.AaLearner import AaLearner

from pprint import pprint
import matplotlib.pyplot as plt
import pickle





class RsLearner(AaLearner):
    """This agent tests out randomly generated augmentation policies.

    Tests randomly sampled policies from the search space specified by the AutoAugment
    paper. Random search has been shown to be a hard baseline to beat for many other
    hyper-parameter optimization tasks (see References below). 
    Hence, this learner acts as a difficult baseline for other AaLearner's.

    Args:
        sp_num (int, optional): number of subpolicies per policy. Defaults to 5.

        p_bins (int, optional): number of bins we divide the interval [0,1] for 
                        probabilities. e.g. (0.0, 0.1, ... 1.0) Defaults to 11.

        m_bins (int, optional): number of bins we divide the magnitude space.
                        Defaults to 10.

        exclude_method (list, optional): list of names(:type:str) of image operations
                        the user wants to exclude from the search space. Defaults to [].

        batch_size (int, optional): child_network training parameter. Defaults to 32.

        toy_size (int, optional): child_network training parameter. ratio of original
                            dataset used in toy dataset. Defaults to 0.1.

        learning_rate (float, optional): child_network training parameter. Defaults to 1e-2.

        max_epochs (Union[int, float], optional): child_network training parameter. 
                            Defaults to float('inf').

        early_stop_num (int, optional): child_network training parameter. Defaults to 20.
    
    Attributes:
        history (list): list of policies that has been input into 
                        self._test_autoaugment_policy as well as their respective obtained
                        accuracies

        augmentation_space (list): list of image functions that the user has chosen to 
                        include in the search space.

    References
    ----------
    Ekin D. Cubuk, et al. 
        "AutoAugment: Learning Augmentation Policies from Data"
        arXiv:1805.09501
    
    Bergstra James, Yoshua Bengio
        "Random Search for Hyper-Parameter Optimization"
        https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

    """
    def __init__(self,
                # parameters that define the search space
                sp_num=5,
                p_bins=11,
                m_bins=10,
                exclude_method=[],
                # hyperparameters for when training the child_network
                batch_size=8,
                toy_size=1,
                learning_rate=1e-1,
                max_epochs=float('inf'),
                early_stop_num=30,
                ):
        
        super().__init__(
                    sp_num=sp_num, 
                    p_bins=p_bins, 
                    m_bins=m_bins, 
                    discrete_p_m=True,
                    batch_size=batch_size,
                    toy_size=toy_size,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    early_stop_num=early_stop_num,
                    exclude_method=exclude_method
                    )
        

    def _generate_new_discrete_operation(self):
        """
        generate a new random operation in the form of a tensor of dimension:
            (fun_num + 11 + 10)

        Used only when self.discrete_p_m=True

        The first fun_num dimensions is a 1-hot encoding to specify which function to use.
        The next 11 dimensions specify which 'probability' to choose.
            (0.0, 0.1, ..., 1.0)
        The next 10 dimensions specify which 'magnitude' to choose.
            (0, 1, ..., 9)
        """

        random_fun = np.random.randint(0, self.fun_num)
        random_prob = np.random.randint(0, self.p_bins)
        random_mag = np.random.randint(0, self.m_bins)
        
        fun_t= torch.zeros(self.fun_num)
        fun_t[random_fun] = 1.0
        prob_t = torch.zeros(self.p_bins)
        prob_t[random_prob] = 1.0
        mag_t = torch.zeros(self.m_bins)
        mag_t[random_mag] = 1.0

        return torch.cat([fun_t, prob_t, mag_t])


    def _generate_new_continuous_operation(self):
        """
        Returns operation_tensor, which is a tensor representation of a random operation with
        dimension:
            (fun_num + 1 + 1)

        Used only when self.discrete_p_m=False.

        The first fun_num dimensions is a 1-hot encoding to specify which function to use.
        The next 1 dimensions specify which 'probability' to choose.
            0 < x < 1
        The next 1 dimensions specify which 'magnitude' to choose.
            0 < x < 9
        """

        fun_p_m = torch.zeros(self.fun_num + 2)
        
        # pick a random image function
        random_fun = np.random.randint(0, self.fun_num)
        fun_p_m[random_fun] = 1

        fun_p_m[-2] = np.random.uniform() # 0<prob<1
        fun_p_m[-1] = np.random.uniform() * (self.m_bins-0.0000001) - 0.4999999 # -0.5<mag<9.5
        
        return fun_p_m


    def _generate_new_policy(self):
        """
        Generates a new policy, with the elements chosen at random
        (unifom random distribution).
        """

        new_policy = []
        
        for _ in range(self.sp_num): # generate sp_num subpolicies for each policy
            ops = []
            # generate 2 operations for each subpolicy
            for i in range(2):
                # if our agent uses discrete representations of probability and magnitude
                if self.discrete_p_m:
                    new_op = self._generate_new_discrete_operation()
                else:
                    new_op = self._generate_new_continuous_operation()
                new_op = self._translate_operation_tensor(new_op)
                ops.append(new_op)

            new_subpolicy = tuple(ops)

            new_policy.append(new_subpolicy)

        return new_policy


    def learn(self, 
            train_dataset, 
            test_dataset, 
            child_network_architecture, 
            iterations=15):
        # test out `iterations` number of  random policies
        for _ in range(iterations):
            policy = self._generate_new_policy()

            pprint(policy)
            reward = self._test_autoaugment_policy(policy,
                                                child_network_architecture,
                                                train_dataset,
                                                test_dataset)
    



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

    agent = RsLearner(
                                sp_num=7,
                                toy_size=0.01,
                                batch_size=4,
                                learning_rate=0.05,
                                max_epochs=float('inf'),
                                early_stop_num=35,
                                )
    agent.learn(train_dataset,
                test_dataset,
                child_network_architecture=child_network_architecture,
                iterations=3)

    # with open('randomsearch_logs.pkl', 'wb') as file:
    #                 pickle.dump(self.history, file)
    pprint(agent.history)