import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms.autoaugment as autoaugment
import MetaAugment.child_networks as child_networks
import random

np.random.seed(0)
random.seed(0)

# code from https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/train.py



# code from https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/train.py
batch_size = 32
n_samples = 0.02

train_dataset = datasets.MNIST(root='./MetaAugment/train', train=True, download=False, transform=torchvision.transforms.ToTensor())
test_dataset = datasets.MNIST(root='./MetaAugment/test', train=False, download=False, transform=torchvision.transforms.ToTensor())
# shuffle and take first n_samples  %age of training dataset
shuffled_train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)).tolist())
indices_train = torch.arange(int(n_samples*len(train_dataset)))
reduced_train_dataset = torch.utils.data.Subset(shuffled_train_dataset, indices_train)
# shuffle and take first n_samples %age of test dataset
shuffled_test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)).tolist())
indices_test = torch.arange(int(n_samples*len(test_dataset)))
reduced_test_dataset = torch.utils.data.Subset(shuffled_test_dataset, indices_test)

print("Size of training dataset:\t", len(reduced_train_dataset))
print("Size of testing dataset:\t", len(reduced_test_dataset), "\n")

train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(reduced_test_dataset, batch_size=batch_size)

model = child_networks.lenet()
sgd = optim.SGD(model.parameters(), lr=1e-1)
cost = nn.CrossEntropyLoss()
epoch = 100

# using torchvision.transforms.autoaugment module
policy = autoaugment.AutoAugmentPolicy("cifar10")
aa = autoaugment.AutoAugment(policy=policy)

def train_model(model, train_loader, test_loader, sgd, cost, epoch):
    # train a network(model) for 'epoch' epochs
    for _epoch in range(epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = cost(predict_y, train_label.long())
            #if idx % 10 == 0:
            #    print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()

        correct = 0
        _sum = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        if _epoch % 10 == 0:
            print('Epoch: {} \t Accuracy: {:.2f}%'.format(_epoch, correct / _sum *100))
        #torch.save(model, f'mnist_{correct / _sum}.pkl')

    performance = 0
    return performance

def prepare_dataset(dataset, policy, toy_flag):
    '''
    takes in dataset and policy, returns train_loader and test_loader.
    toy_flag: whether or not we should return a toy dataset
    '''


class AA_Learner:
    def __init__(self, controller):
        self.controller = controller

    def learn(self, dataset, child_network, toy_flag):
        '''
        Deos what is seen in Figure 1 in the AutoAugment paper.

        'res' stands for resolution of the discretisation of the search space. It could be
        a tuple, with first entry regarding probability, second regarding magnitude
        '''
        good_policy_found = False

        while not good_policy_found:
            policy = self.controller.pop_policy()

            train_loader, test_loader = prepare_dataset(dataset, policy, toy_flag)

            reward = train_model(child_network, train_loader, test_loader, sgd, cost, epoch)

            self.controller.update(reward, policy)
        
        return good_policy


