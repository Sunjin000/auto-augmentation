import torch

import MetaAugment.child_networks as cn
from MetaAugment.autoaugment_learners.aa_learner import aa_learner

from pprint import pprint
import pickle


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
                # evolutionary learner specific settings
                ):