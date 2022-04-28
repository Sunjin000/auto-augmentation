# We can initialize the train_dataset with its transform as None.
# Later on, we will change this object's transform attribute to the policy
# that we want to test
import torchvision.datasets as datasets
import torchvision

import MetaAugment.child_networks as cn
from MetaAugment.autoaugment_learners.AaLearner import AaLearner
from MetaAugment.autoaugment_learners.gen_learner import Genetic_learner

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
                            )


agent.learn(train_dataset,
            test_dataset,
            child_network_architecture=child_network_architecture,
            iterations=10)

# with open('randomsearch_logs.pkl', 'wb') as file:
#                 pickle.dump(self.history, file)
print(agent.history)