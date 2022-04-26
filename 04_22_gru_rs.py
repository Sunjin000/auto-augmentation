import torchvision.datasets as datasets
import torchvision
import torch

import meta_augment.child_networks as cn
import meta_augment.autoaugment_learners as aal

from pathlib import Path

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
    total_iter,
    train_dataset,
    test_dataset,
    child_network_architecture,
    agent_arch,
    config,
    ):
    try:
        # try to load agent
        with open(save_file, 'rb') as f:
            agent = torch.load(f, map_location=device)
    except (FileNotFoundError, RuntimeError):
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
                    iterations=5
                    )
        # save agent every iteration
        with open(save_file, 'wb+') as f:
            torch.save(agent, f)
    
    print("FINAL POLICIES: ", agent_arch.policy_result)

    print('run_benchmark closing')


# aa_learner config
controller = cn.LeNet(img_height=28, img_width=28, num_labels=16*2, img_channels=1)


# aa_learner config
# config = {
#         'sp_num' : 5,
#         'learning_rate' : 1e-1,
#         'toy_flag' : False,
# #         'toy_flag' : True,
# #         'toy_size' : 0.001,
#         'batch_size' : 32,
#         'max_epochs' : 100,
#         'early_stop_num' : 10,
#         'controller' : controller,
#         'num_solutions' : 10,
#         }
# total_iter=150


# # FashionMNIST with SimpleNet
# train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
#                             train=True, download=True, transform=torchvision.transforms.ToTensor())
# test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
#                         train=False, download=True,
#                         transform=torchvision.transforms.ToTensor())
# child_network_architecture = cn.SimpleNet


# run_benchmark(
#     save_file='/bench_test/04_22_fm_sn_gru.pkl',
#     total_iter=total_iter,
#     train_dataset=train_dataset,
#     test_dataset=test_dataset,
#     child_network_architecture=child_network_architecture,
#     agent_arch=aal.evo_learner,
#     config=config,
#     )







# CIFAR10 with LeNet
train_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=False, download=True, 
                        transform=torchvision.transforms.ToTensor())
child_network_architecture = cn.LeNet(img_height = 32, 
                                      img_width=32, 
                                      num_labels=10, 
                                      img_channels=3)




controller = cn.LeNet(img_height=32, img_width=32, num_labels=16*2, img_channels=3)
config = {
        'sp_num' : 5,
        'learning_rate' : 1e-1,
#         'toy_flag' : True,
#         'toy_size' : 0.001,
        'batch_size' : 32,
        'max_epochs' : 100,
        'early_stop_num' : 10,
        'controller' : controller,
        }
total_iter=150

# # gru
run_benchmark(
    save_file='bench_test/04_22_cf_ln_evo',
    total_iter=total_iter,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    child_network_architecture=child_network_architecture,
    agent_arch=aal.evo_learner,
    config=config,
    )


megapol = [(('ShearY', 0.5, 5), ('Posterize', 0.6, 5)), (('Color', 1.0, 9), ('Contrast', 1.0, 9)), (('TranslateX', 0.5, 5), ('Posterize', 0.5, 5)), (('TranslateX', 0.5, 5), ('Posterize', 0.5, 5)), (('Color', 0.5, 5), ('Posterize', 0.5, 5))]


# accs=[]
# for _ in range(10):
#     print(f'{_}/{10}')
#     temp_agent = aal.evo_learner(**config)
#     accs.append(
#             temp_agent.test_autoaugment_policy(megapol,
#                                 child_network_architecture,
#                                 train_dataset,
#                                 test_dataset,
#                                 logging=False)
#                 )

# print("FASION MNIST accs: ", accs)
