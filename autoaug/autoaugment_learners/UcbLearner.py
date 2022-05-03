import numpy as np

from tqdm import trange

from ..child_networks import *
from .RsLearner import RsLearner


class UcbLearner(RsLearner):
    """
    Uses the UCB1 algorithm originally developed for multi-armed bandit problems.
    Recommended when 

    - Optimal hyperparameters for training the CNN (such as batch size and learning rate) has not been found yet.

    - Using a toy dataset or a toy CNN

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
    
        num_policies (int, optional): Number of policies we want to serach over. 
                            Defaults to 100.
        
    Attributes:
        history (list): list of policies that has been input into 
                        self._test_autoaugment_policy as well as their respective obtained
                        accuracies

        augmentation_space (list): list of image functions that the user has chosen to 
                        include in the search space.

        policies (list): A list of policies which we are currently searching over.

        avg_accs (list): A list where the nth element indicates the average accuracy 
                        obtained by the nth policy.



    Notes
    -----
    As opposed the the other learners, this searches over a subset of the entire
    search space (specified in the AutoAugment paper). The size of the subset is 
    initialized to be ``self.num_policies``. But we can increase it by running 
    self.make_more_policies(). For example, we initialize the learner with 
    ``self.num_policies=7``, run ``self.learn(iterations=20)`` to learn about the
    seven policies we have in our ``self.policies``. Then run 
    ``self.make_more_policies(n=5)`` to add 5 more policies to ``self.policies``.
    Then we can run ``self.learn(iterations=20)`` to continue the UCB1 algorithm
    with the extended search space.

    References
    ----------
    Peter Auer, et al.
        "Finite-time Analysis of the Multiarmed Bandit Problem"
        https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    
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
                # UcbLearner specific hyperparameter
                num_policies=100
                ):
        
        super().__init__(
                        sp_num=sp_num, 
                        p_bins=p_bins, 
                        m_bins=m_bins, 
                        batch_size=batch_size,
                        toy_size=toy_size,
                        learning_rate=learning_rate,
                        max_epochs=max_epochs,
                        early_stop_num=early_stop_num,
                        exclude_method=exclude_method,
                        )
        

        

        # attributes used in the UCB1 algorithm
        self.num_policies = num_policies

        self.policies = [self._generate_new_policy() for _ in range(num_policies)]

        self.avg_accs = [None]*self.num_policies
        self.best_avg_accs = []

        self.cnts = [0]*self.num_policies
        self.q_plus_cnt = [0]*self.num_policies
        self.total_count = 0




    def make_more_policies(self, n):
        """generates n more random policies and adds it to self.policies

        Args:
            n (int): how many more policies to we want to randomly generate
                    and add to our list of policies
        """

        self.policies += [self._generate_new_policy() for _ in range(n)]

        # all the below need to be lengthened to store information for the 
        # new policies
        self.avg_accs += [None for _ in range(n)]
        self.cnts += [0 for _ in range(n)]
        self.q_plus_cnt += [None for _ in range(n)]
        self.num_policies += n



    def learn(self, 
            train_dataset, 
            test_dataset, 
            child_network_architecture, 
            iterations=15,):
        """continue the UCB algorithm for ``iterations`` number of turns

        """

        for this_iter in trange(iterations):
            print("iter: ", this_iter)
            if this_iter > 5:
                print("best 5 pols: ", self.get_n_best_policies(5))
            # choose which policy we want to test
            if None in self.avg_accs:
                # if there is a policy we haven't tested yet, we 
                # test that one
                this_policy_idx = self.avg_accs.index(None)
                this_policy = self.policies[this_policy_idx]
                acc = self._test_autoaugment_policy(
                                this_policy,
                                child_network_architecture,
                                train_dataset,
                                test_dataset,
                                logging=False,
                                )
                # update q_values (average accuracy)
                self.avg_accs[this_policy_idx] = acc
            else:
                # if we have tested all policies before, we test the
                # one with the best q_plus_cnt value
                this_policy_idx = np.argmax(self.q_plus_cnt)
                this_policy = self.policies[this_policy_idx]
                acc = self._test_autoaugment_policy(
                                this_policy,
                                child_network_architecture,
                                train_dataset,
                                test_dataset,
                                logging=False,
                                )
                # update q_values (average accuracy)
                self.avg_accs[this_policy_idx] = (self.avg_accs[this_policy_idx]*self.cnts[this_policy_idx] + acc) / (self.cnts[this_policy_idx] + 1)
    
            # logging the best avg acc up to now
            best_avg_acc = max([x for x in self.avg_accs if x is not None])
            self.best_avg_accs.append(best_avg_acc)

            # print progress for user
            if (this_iter+1) % 5 == 0:
                print("Iteration: {},\tQ-Values: {}, Best this_iter: {}".format(
                                this_iter+1, 
                                list(np.around(np.array(self.avg_accs),2)), 
                                max(list(np.around(np.array(self.avg_accs),2)))
                                )
                    )

            # update counts
            self.cnts[this_policy_idx] += 1
            self.total_count += 1

            # update q_plus_cnt values every turn after the initial sweep through
            for i in range(self.num_policies):
                if self.avg_accs[i] is not None:
                    self.q_plus_cnt[i] = self.avg_accs[i] + np.sqrt(2*np.log(self.total_count)/self.cnts[i])
            
            print(self.cnts)


            
    def get_mega_policy(self, number_policies=5):
        """
        Produces a mega policy, based on the n best subpolicies (evo learner)/policies
        (other learners)

        
        Args: 
            number_policies -> int: Number of (sub)policies to be included in the mega
            policy

        Returns:
            megapolicy -> [subpolicy, subpolicy, ...]
        """

        temp_avg_accs = [x if x is not None  else 0 for x in self.avg_accs]

        temp_history = list(zip(self.policies, temp_avg_accs))

        number_policies = max(number_policies, len(temp_history))

        inter_pol = sorted(temp_history, key=lambda x: x[1], reverse = True)[:number_policies]

        megapol = []
        for pol in inter_pol:
            megapol += pol[0]

        return megapol


    def get_n_best_policies(self, number_policies=5):
        """
        returns the n best policies

        
        Args: 
            number_policies (int): Number of (sub)policies to return

        Returns:
            list of best n policies
        """

        temp_avg_accs = [x if x is not None  else 0 for x in self.avg_accs]

        temp_history = list(zip(self.policies, temp_avg_accs))

        number_policies = max(number_policies, len(temp_history))

        inter_pol = sorted(temp_history, key=lambda x: x[1], reverse = True)[:number_policies]

        return inter_pol[number_policies]


       




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