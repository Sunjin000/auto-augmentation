# We can initialize the train_dataset with its transform as None.
# Later on, we will change this object's transform attribute to the policy
# that we want to test
import torchvision.datasets as datasets
import torchvision

import autoaug.child_networks as cn
from autoaug.autoaugment_learners.AaLearner import AaLearner
from autoaug.autoaugment_learners.GenLearner import Genetic_learner

import random
    
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

agent = Genetic_learner(
                            sp_num=2,
                            toy_size=0.01,
                            batch_size=4,
                            learning_rate=0.05,
                            max_epochs=float('inf'),
                            early_stop_num=10,
                            num_offspring=10
                            )


agent.learn(train_dataset,
            test_dataset,
            child_network_architecture=child_network_architecture,
            iterations=100)

# with open('genetic_logs.pkl', 'wb') as file:
#                 pickle.dump(agent.history, file)
print(sorted(agent.history, key = lambda x: x[1]))