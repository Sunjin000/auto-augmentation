import numpy as np

from tqdm import trange

from ..child_networks import *
from .randomsearch_learner import randomsearch_learner


class ucb_learner(randomsearch_learner):
    """
    Tests randomly sampled policies from the search space specified by the AutoAugment
    paper. Acts as a baseline for other aa_learner's.
    """
    def __init__(self,
                # parameters that define the search space
                sp_num=5,
                p_bins=11,
                m_bins=10,
                discrete_p_m=True,
                exclude_method=[],
                # hyperparameters for when training the child_network
                batch_size=8,
                toy_size=1,
                learning_rate=1e-1,
                max_epochs=float('inf'),
                early_stop_num=30,
                # ucb_learner specific hyperparameter
                num_policies=100
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
            iterations=15,
            print_every_epoch=False):
        """continue the UCB algorithm for `iterations` number of turns

        """

        for this_iter in trange(iterations):

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
                                print_every_epoch=print_every_epoch
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
                                print_every_epoch=print_every_epoch
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