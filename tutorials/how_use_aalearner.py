# You can run this in the main directory by typing:
# python -m tutorials.how_use_aalearner

import MetaAugment.autoaugment_learners as aal
import MetaAugment.child_networks as cn

import torchvision.datasets as datasets
import torchvision


# Defining our problem setting:
# In other words, specifying the dataset and the child network
train_dataset = datasets.MNIST(root='./MetaAugment/datasets/mnist/train',
                                train=True, download=True, transform=None)
test_dataset = datasets.MNIST(root='./MetaAugment/datasets/mnist/test', 
                        train=False, download=True, transform=torchvision.transforms.ToTensor())
child_network = cn.lenet


# NOTE: It is important not to type:
#   child_network = cn.lenet()
# We need the ``child_network`` variable to be a ``type``, not a ``nn.Module``
# because the ``child_network`` will be called multiple times to initialize a 
# ``nn.Module`` of its architecture multiple times: once every time we need to 
# train a different network to evaluate a different policy.


# Using the random search learner to evaluate randomly generated policies
rs_agent = aal.randomsearch_learner()
rs_agent.learn(train_dataset, test_dataset, child_network, toy_flag=True)


# Viewing the results
# ``.history`` is a list containing all the policies tested and the respective
# accuracies obtained when trained using them
print(rs_agent.history)