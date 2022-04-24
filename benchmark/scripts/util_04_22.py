from matplotlib.pyplot import get
import torchvision.datasets as datasets
import torchvision
import torch

import MetaAugment.child_networks as cn
import MetaAugment.autoaugment_learners as aal

from pprint import pprint

"""
testing gru_learner and randomsearch_learner on

  fashionmnist with simple net

 and 

  cifar10 with lenet

"""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def run_benchmark(
    save_file,
    train_dataset,
    test_dataset,
    child_network_architecture,
    agent_arch,
    config,
    total_iter=150,
    ):
    try:
        # try to load agent
        with open(save_file, 'rb') as f:
            agent = torch.load(f, map_location=device)
    except FileNotFoundError:
        # if agent hasn't been saved yet, initialize the agent
        agent = agent_arch(**config)


    # if history is not length total_iter yet(if total_iter
    # different policies haven't been tested yet), keep running
    while len(agent.history)<total_iter:
        print(f'{len(agent.history)} / {total_iter}')
        # run 1 iteration (test one new policy and update the GRU)
        agent.learn(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    child_network_architecture=child_network_architecture,
                    iterations=1
                    )
        # save agent every iteration
        with open(save_file, 'wb+') as f:
            torch.save(agent, f)

    print('run_benchmark closing')


def get_mega_policy(history, n):
        """
        we get the best n policies from an agent's history,
        concatenate them to form our best mega policy

        Args:
            history (list[tuple])
            n (int)
        
        Returns:
            list[float]: validation accuracies
        """
        assert len(history) >= n

        # agent.history is a list of (policy(list), val_accuracy(float)) tuples 
        sorted_history = sorted(history, key=lambda x:x[1]) # sort wrt acc

        best_history = sorted_history[:n]

        megapolicy = []
        for policy,acc in best_history:
            for subpolicy in policy:
                megapolicy.append(subpolicy)
        
        return megapolicy


def rerun_best_policy(
    agent_pickle,
    accs_txt,
    train_dataset,
    test_dataset,
    child_network_architecture,
    repeat_num
    ):

    with open(agent_pickle, 'rb') as f:
        agent = torch.load(f, map_location=device)
    
    megapol = get_mega_policy(agent.history)
    print('mega policy to be tested:')
    pprint(megapol)
    
    accs=[]
    for _ in range(repeat_num):
        print(f'{_}/{repeat_num}')
        accs.append(
                agent.test_autoaugment_policy(megapol,
                                    child_network_architecture,
                                    train_dataset,
                                    test_dataset,
                                    logging=False)
                    )
        with open(accs_txt, 'w') as f:
            f.write(str(accs))
