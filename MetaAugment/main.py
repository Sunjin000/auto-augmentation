import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms.autoaugment as autoaugment
#import MetaAugment.AutoAugmentDemo.ops as ops # 

# code from https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/train.py



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


def train_child_network(child_network, train_loader, test_loader, sgd,
                         cost, max_epochs=100, early_stop_num = 10):
    best_acc=0
    early_stop_cnt = 0
    
    # train child_network and check validation accuracy each epoch
    for _epoch in range(max_epochs):

        # train child_network
        child_network.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
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
        
        print('main.train_child_network best accuracy: ', best_acc)
        
    return best_acc

# This is sort of how our AA_Learner class should look like:
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