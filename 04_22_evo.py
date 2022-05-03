import torchvision.datasets as datasets
import torchvision
import torch

import autoaug.child_networks as cn
import autoaug.autoaugment_learners as aal


controller = cn.EasyNet(img_height=28, img_width=28, num_labels=16*2, img_channels=1)




# config = {
        # 'sp_num' : 5,
        # 'learning_rate' : 1e-1,
        # 'batch_size' : 32,
        # 'max_epochs' : 100,
        # 'early_stop_num' : 10,
        # 'controller' : controller,
        # 'num_solutions' : 10,
        # }


# config = {
#         'sp_num' : 5,
#         'learning_rate' : 1e-1,
#         'batch_size' : 32,
#         'max_epochs' : 100,
#         'early_stop_num' : 10,
#         'num_offspring' : 10,
#         }


config = {
        'sp_num' : 5,
        'learning_rate' : 1e-1,
        'batch_size' : 32,
        'max_epochs' : 100,
        'early_stop_num' : 10,
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
    
    while len(agent.history)<total_iter:
        print(f'{len(agent.history)} / {total_iter}')
        # run 1 iteration (test one new policy and update the GRU)
        agent.learn(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    child_network_architecture=child_network_architecture,
                    iterations=1
                    )
        print("pols: ", agent.history)
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




# # # CIFAR10 with LeNet
train_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=True, download=True, transform=None)
test_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                        train=False, download=True, 
                        transform=torchvision.transforms.ToTensor())



# train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
#                             train=True, download=True, transform=torchvision.transforms.ToTensor())
# test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
#                         train=False, download=True,
#                         transform=torchvision.transforms.ToTensor())




child_network_architecture = cn.LeNet(
                                    img_height=32,
                                    img_width=32,
                                    num_labels=10,
                                    img_channels=3
                                    )


# child_network_architecture = cn.EasyNet(
#                                     img_height=28,
#                                     img_width=28,
#                                     num_labels=10,
#                                     img_channels=1
#                                     )


save_dir='./benchmark/pickles/04_22_cf_ln_rssadasdsad'

# evo
# run_benchmark(
#     save_file=save_dir+'.pkl',
#     train_dataset=train_dataset,
#     test_dataset=test_dataset,
#     child_network_architecture=child_network_architecture,
#     agent_arch=aal.UcbLearner,
#     # agent_arch=aal.GenLearner,
#     config=config,
#     )




# megapol = [(('ShearY', 0.5, 5), ('Posterize', 0.6, 5)), (('Color', 1.0, 9), ('Contrast', 1.0, 9)), (('TranslateX', 0.5, 5), ('Posterize', 0.5, 5)), (('TranslateX', 0.5, 5), ('Posterize', 0.5, 5)), (('Color', 0.5, 5), ('Posterize', 0.5, 5))]

# Evo learner CIPHAR:  [0.6046000123023987, 0.6050999760627747, 0.5861999988555908, 0.5936999917030334, 0.5949000120162964, 0.5791000127792358, 0.6000999808311462, 0.6017000079154968, 0.5983999967575073, 0.5885999798774719]
# megapol = [(('Equalize', 0.5, None), ('TranslateX', 0.5, 9)), (('Equalize', 0.5, None), ('TranslateX', 0.5, 8)), (('TranslateY', 0.5, 6), ('Brightness', 0.5, 6)), (('ShearY', 0.9, 5), ('Rotate', 0.5, 5)), (('TranslateX', 0.6, 5), ('Color', 1.0, 5))]
# megapol = [(('TranslateX', 0.3, 2), ('Sharpness', 0.4, 2)), (('Color', 0.4, 8), ('Contrast', 0.5, 1)), (('Sharpness', 0.0, 7), ('ShearY', 0.6, 2)), (('Equalize', 0.1, None), ('ShearY', 1.0, 6)), (('Solarize', 0.1, 5), ('Color', 0.9, 3))]


# Genetic learner FASHION   [0.8870999813079834, 0.8906000256538391, 0.8853999972343445, 0.8866000175476074, 0.8924000263214111, 0.8889999985694885, 0.8859999775886536, 0.8910999894142151, 0.8871999979019165, 0.8848000168800354]
# megapol = [(('Brightness', 0.6, 1), ('Color', 0.2, 9)), (('Brightness', 0.6, 7), ('Color', 0.2, 9)), (('AutoContrast', 0.9, None), ('Invert', 0.0, None)), (('Sharpness', 0.9, 3), ('AutoContrast', 0.9, None)), (('Brightness', 0.6, 3), ('Color', 0.2, 9))]


# UCB1
# megapol = [(('TranslateY', 0.5, 7), ('Invert', 0.6, None)), (('Solarize', 0.8, 7), ('Color', 0.2, 9)), (('Invert', 0.8, None), ('Contrast', 0.5, 8)), (('Posterize', 0.8, 0), ('Rotate', 0.0, 8)), (('Equalize', 0.7, None), ('TranslateY', 0.5, 1))]

megapol = [(('Rotate', 0.6, 5), ('ShearX', 0.1, 2)), (('Sharpness', 0.2, 5), ('Invert', 0.6, None)), (('ShearY', 0.7, 9), ('Sharpness', 0.1, 0)), (('Contrast', 0.5, 0), ('AutoContrast', 1.0, None)), (('Solarize', 0.7, 6), ('Posterize', 0.8, 5))]


# megapol = [(('Brightness', 0.0, 1), ('Color', 0.0, 9))]


accs=[]
for _ in range(10):
    print("CIFAR BASELINE UCB ONE")
    print(f'{_}/{10}')
    # temp_agent = aal.EvoLearner(**config)
    temp_agent = aal.GenLearner(**config)
    accs.append(
            temp_agent._test_autoaugment_policy(megapol,
                                child_network_architecture,
                                train_dataset,
                                test_dataset,
                                logging=False)
                )

print("CIPHAR10 accs: ", accs)
