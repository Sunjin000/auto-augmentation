# %%
import numpy as np
import matplotlib.pyplot as plt 
from itertools import count

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader


from collections import namedtuple, deque
import math
import random

from MetaAugment.main import *


batch_size = 128

test_dataset = torchvision.datasets.MNIST(root='test_dataset/', train=False, download=True, transform=torchvision.transforms.ToTensor())
train_dataset = torchvision.datasets.MNIST(root='test_dataset/', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print('test_loader', len(test_loader))
print('train_loader',len(train_loader))

def create_toy(train_dataset, test_dataset, batch_size, n_samples):
    # shuffle and take first n_samples %age of training dataset
    shuffled_train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)).tolist())
    indices_train = torch.arange(int(n_samples*len(train_dataset)))
    reduced_train_dataset = torch.utils.data.Subset(shuffled_train_dataset, indices_train)
    # shuffle and take first n_samples %age of test dataset
    shuffled_test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)).tolist())
    indices_test = torch.arange(int(n_samples*len(test_dataset)))
    reduced_test_dataset = torch.utils.data.Subset(shuffled_test_dataset, indices_test)

    # push into DataLoader
    train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(reduced_test_dataset, batch_size=batch_size)

    return train_loader, test_loader

# train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size, 10)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

# %% [markdown]
# ## collect reward

# %%

def collect_reward(train_loader, test_loader, max_epochs=100, early_stop_num=10):
    child_network = LeNet() 
    sgd = optim.SGD(child_network.parameters(), lr=1e-1)
    cost = nn.CrossEntropyLoss()
    best_acc=0
    early_stop_cnt = 0
    
    # train child_network and check validation accuracy each epoch
    print('max_epochs', max_epochs)
    for _epoch in range(max_epochs):
        print('_epoch', _epoch)
        # train child_network
        child_network.train()
        for t, (train_x, train_label) in enumerate(train_loader):
            label_np = np.zeros((train_label.shape[0], 10))
            sgd.zero_grad()
            predict_y = child_network(train_x.float())
            loss = cost(predict_y, train_label.long())
            loss.backward()
            sgd.step()

        # check validation accuracy on validation set
        correct = 0
        _sum = 0
        child_network.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = child_network(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            label_np = test_label.numpy()
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]
        
        # update best validation accuracy if it was higher, otherwise increase early stop count
        acc = correct / _sum

        if acc > best_acc :
            best_acc = acc
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        # exit if validation gets worse over 10 runs
        if early_stop_cnt >= early_stop_num:
            break

        # if _epoch%30 == 0:
        #     print('child_network accuracy: ', best_acc)
        
    return best_acc


# %%
for t, (train_x, train_label) in enumerate(test_loader):
    print(train_x.shape)
    print(train_label)
    break
len(test_loader)

# %%
collect_reward(train_loader, test_loader)


# %% [markdown]
# ## Policy network

# %%
class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5 , stride=2)
        self.conv2 = nn.Conv2d(6, 12, 5, stride=2)
        self.maxpool = nn.MaxPool2d(4)

        # actor's layer
        self.action_head = nn.Linear(12, 2)

        # critic's layer
        self.value_head = nn.Linear(12, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        # print('x', x.shape)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        # print('self.action_head(x)', self.action_head(x).shape)
        action_prob = F.softmax(self.action_head(x), dim=-1)
        # print('action_prob', action_prob.shape)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values


# %%
test_dataset = torchvision.datasets.MNIST(root='test_dataset/', train=False, download=True, transform=torchvision.transforms.ToTensor())
train_dataset = torchvision.datasets.MNIST(root='test_dataset/', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

policy_model = Policy()
# for t, (x, y) in enumerate(train_loader):
#     # print(x.shape)
#     policy_model(x)

# %% [markdown]
# ## select action

# %%
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def select_action(train_loader, policy_model):
    probs_list = []
    value_list = []
    for t, (x, y) in enumerate(train_loader):
        probs_i, state_value_i = policy_model(x)
        probs_list += [probs_i]
        value_list += [state_value_i]

    probs = torch.mean(torch.cat(probs_list), dim=0)
    state_value = torch.mean(torch.cat(value_list))
    # print('probs_i', probs_i)
    # print('probs', probs)
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)
    # print('m', m)
    # and sample an action using the distribution
    action = m.sample()
    # print('action', action)

    # save to action buffer
    policy_model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


# %%
torch.tensor([1, 2, 3])

# %% [markdown]
# ## take action

# %%
def take_action(action_idx):
    # Define actions (data augmentation policy) --- can be improved
    action_list = [
    torchvision.transforms.Compose([torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor()]),
    torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()]),
    torchvision.transforms.Compose([torchvision.transforms.RandomGrayscale(),
        torchvision.transforms.ToTensor()]),
    torchvision.transforms.Compose([torchvision.transforms.RandomAffine(30),
        torchvision.transforms.ToTensor()])]

    # transform   
    transform = action_list[action_idx]
    test_dataset = torchvision.datasets.MNIST(root='test_dataset/', train=False, download=True, transform=transform)
    train_dataset = torchvision.datasets.MNIST(root='test_dataset/', train=True, download=True, transform=transform)
    train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size, n_samples=0.0002)
    return train_loader, test_loader


# %% [markdown]
# ## finish episode

# %%
policy_model = Policy()
optimizer = optim.Adam(policy_model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()
gamma = 0.9
def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = policy_model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in policy_model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del policy_model.rewards[:]
    del policy_model.saved_actions[:]

# %% [markdown]
# ## run

# %%

running_reward = 10
episodes_num = 100
policy_model = Policy()
for i_episode in range(episodes_num) :
    # initiate a new state
    train_dataset = torchvision.datasets.MNIST(root='test_dataset/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    # train_dataset = torchvision.datasets.MNIST(root='test_dataset/', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_loader_state = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # select action from policy
    action_idx = select_action(train_loader, policy_model)
    print('>>> action_idx', action_idx)

    # take the action -> apply data augmentation
    train_loader, test_loader = take_action(action_idx)
    reward = collect_reward(train_loader, test_loader)
    print('>>> reward', reward)

    # if args.render:
    #     env.render()

    policy_model.rewards.append(reward)

    # perform backprop
    finish_episode()

    # # log result
    if i_episode % 10 == 0:
        print('Episode {}\tLast reward (val accuracy): {:.2f}'.format(i_episode, reward))

# %%



