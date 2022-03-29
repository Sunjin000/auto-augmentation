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



def create_toy(train_dataset, test_dataset, batch_size, n_samples, seed=100):
    # shuffle and take first n_samples %age of training dataset
    shuffle_order_train = np.random.RandomState(seed=seed).permutation(len(train_dataset))
    shuffled_train_dataset = torch.utils.data.Subset(train_dataset, shuffle_order_train)
    
    indices_train = torch.arange(int(n_samples*len(train_dataset)))
    reduced_train_dataset = torch.utils.data.Subset(shuffled_train_dataset, indices_train)
    
    # shuffle and take first n_samples %age of test dataset
    shuffle_order_test = np.random.RandomState(seed=seed).permutation(len(test_dataset))
    shuffled_test_dataset = torch.utils.data.Subset(test_dataset, shuffle_order_test)

    indices_test = torch.arange(int(n_samples*len(test_dataset)))
    reduced_test_dataset = torch.utils.data.Subset(shuffled_test_dataset, indices_test)

    # push into DataLoader
    train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(reduced_test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def train_child_network(child_network, train_loader, test_loader, sgd,
                         cost, max_epochs=100, early_stop_num = 10, logging=False):
    best_acc=0
    early_stop_cnt = 0
    
    # logging accuracy for plotting
    acc_log = [] 

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
        with torch.no_grad():
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
        acc_log.append(acc)

    if logging:
        return best_acc, acc_log
    return best_acc

if __name__=='__main__':
    import MetaAugment.child_networks as cn

    batch_size = 32
    n_samples = 0.005

    train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, download=False, 
                                   transform=torchvision.transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, download=False,
                                  transform=torchvision.transforms.ToTensor())

    # create toy dataset from above uploaded data
    train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size, 0.01)

    child_network = cn.lenet()
    sgd = optim.SGD(child_network.parameters(), lr=1e-1)
    cost = nn.CrossEntropyLoss()
    epoch = 20

    best_acc = train_child_network(child_network, train_loader, test_loader, sgd, cost, max_epochs=100)