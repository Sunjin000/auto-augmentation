import torchvision.datasets as datasets
import torchvision
import torch

import MetaAugment.child_networks as cn
import MetaAugment.autoaugment_learners as aal



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