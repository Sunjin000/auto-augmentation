import torchvision.datasets as datasets
import torchvision
import torch

import autoaug.child_networks as cn
import autoaug.autoaugment_learners as aal


controller = cn.EasyNet(img_height=32, img_width=32, num_labels=16*10, img_channels=3)

config = {
        'sp_num' : 5,
        'learning_rate' : 1e-1,
        'batch_size' : 32,
        'max_epochs' : 100,
        'early_stop_num' : 10,
        'controller' : controller,
        'num_solutions' : 10,
        }




import torch

import autoaug.autoaugment_learners as aal

import pprint

"""
testing GruLearner and RsLearner on

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
    
    print("agent history: ", agent.history)
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
        sorted_history = sorted(history, key=lambda x:x[1], reverse=True) # sort wrt acc

        best_history = sorted_history[:n]

        megapolicy = []
        # we also want to keep track of how good the best policies were
        # maybe if we add them all up, they'll become worse! Hopefully better tho
        orig_accs = []

        for policy,acc in best_history:
            for subpolicy in policy:
                megapolicy.append(subpolicy)
            orig_accs.append(acc)
        
        return megapolicy, orig_accs


def rerun_best_policy(
    agent_pickle,
    accs_txt,
    train_dataset,
    test_dataset,
    child_network_architecture,
    config,
    repeat_num
    ):

    with open(agent_pickle, 'rb') as f:
        agent = torch.load(f)
    
    megapol, orig_accs = get_mega_policy(agent.history,3)
    print('mega policy to be tested:')
    pprint.pprint(megapol)
    print(orig_accs)

    accs=[]
    for _ in range(repeat_num):
        print(f'{_}/{repeat_num}')
        temp_agent = aal.AaLearner(**config)
        accs.append(
                temp_agent._test_autoaugment_policy(megapol,
                                    child_network_architecture,
                                    train_dataset,
                                    test_dataset,
                                    logging=False)
                    )
        with open(accs_txt, 'w') as f:
            f.write(pprint.pformat(megapol))
            f.write(str(accs))
            f.write(f'original small policys accuracies: {orig_accs}')




# # CIFAR10 with LeNet
train_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=True, download=True, transform=None)
test_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=False, download=True, 
                        transform=torchvision.transforms.ToTensor())
child_network_architecture = cn.LeNet(
                                    img_height=32,
                                    img_width=32,
                                    num_labels=10,
                                    img_channels=3
                                    )

# save_dir='./benchmark/pickles/04_22_cf_ln_rssad'

# # evo
# run_benchmark(
#     save_file=save_dir+'.pkl',
#     train_dataset=train_dataset,
#     test_dataset=test_dataset,
#     child_network_architecture=child_network_architecture,
#     agent_arch=aal.EvoLearner,
#     config=config,
#     )

# # rerun_best_policy(
# #     agent_pickle=save_dir+'.pkl',
# #     accs_txt=save_dir+'.txt',
# #     train_dataset=train_dataset,
# #     test_dataset=test_dataset,
# #     child_network_architecture=child_network_architecture,
# #     config=config,
# #     repeat_num=5
# #     )



# megapol = [(('ShearY', 0.5, 5), ('Posterize', 0.6, 5)), (('Color', 1.0, 9), ('Contrast', 1.0, 9)), (('TranslateX', 0.5, 5), ('Posterize', 0.5, 5)), (('TranslateX', 0.5, 5), ('Posterize', 0.5, 5)), (('Color', 0.5, 5), ('Posterize', 0.5, 5))]

megapol = [(('Equalize', 0.5, None), ('TranslateX', 0.5, 9)), (('Equalize', 0.5, None), ('TranslateX', 0.5, 8)), (('TranslateY', 0.5, 6), ('Brightness', 0.5, 6)), (('ShearY', 0.9, 5), ('Rotate', 0.5, 5)), (('TranslateX', 0.6, 5), ('Color', 1.0, 5))]



accs=[]
for _ in range(10):
    print(f'{_}/{10}')
    temp_agent = aal.EvoLearner(**config)
    accs.append(
            temp_agent._test_autoaugment_policy(megapol,
                                child_network_architecture,
                                train_dataset,
                                test_dataset,
                                logging=False)
                )

print("CIPHAR10 accs: ", accs)
