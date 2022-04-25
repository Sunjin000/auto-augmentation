import MetaAugment.autoaugment_learners as aal
import MetaAugment.child_networks as cn
import torch
import torchvision
import torchvision.datasets as datasets

import random


def test_ucb_learner():
    policies = UCB1_JC.generate_policies(num_policies, num_sub_policies)
        q_values, best_q_values = UCB1_JC.run_UCB1(
                                                policies,
                                                batch_size, 
                                                learning_rate, 
                                                ds, 
                                                toy_size, 
                                                max_epochs, 
                                                early_stop_num, 
                                                iterations, 
                                                IsLeNet, 
                                                ds_name
                                                )     
        best_q_values = np.array(best_q_values)
    pass