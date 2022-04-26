from MetaAugment.child_networks import *
from MetaAugment.main import create_toy, train_child_network
import torch
import torchvision
import torchvision.datasets as datasets
import pickle

def parse_ds_cn_arch(ds, ds_name, IsLeNet):
    if ds == "MNIST":
        train_dataset = datasets.MNIST(root='./datasets/mnist/train', 
                        train=True, download=True, transform=None)
        test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, 
                        download=True, transform=torchvision.transforms.ToTensor())
    elif ds == "KMNIST":
        train_dataset = datasets.KMNIST(root='./datasets/kmnist/train', 
                        train=True, download=True, transform=None)
        test_dataset = datasets.KMNIST(root='./datasets/kmnist/test', train=False, 
                        download=True, transform=torchvision.transforms.ToTensor())
    elif ds == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train', 
                        train=True, download=True, transform=None)
        test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', train=False, 
                        download=True, transform=torchvision.transforms.ToTensor())
    elif ds == "CIFAR10":
        train_dataset = datasets.CIFAR10(root='./datasets/cifar10/train', 
                        train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR10(root='./datasets/cifar10/test', train=False, 
                        download=True, transform=torchvision.transforms.ToTensor())
    elif ds == "CIFAR100":
        train_dataset = datasets.CIFAR100(root='./datasets/cifar100/train', 
                        train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR100(root='./datasets/cifar100/test', train=False, 
                        download=True, transform=torchvision.transforms.ToTensor())
    elif ds == 'Other':
        dataset = datasets.ImageFolder('./datasets/upload_dataset/'+ ds_name, transform=None)
        len_train = int(0.8*len(dataset))
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len_train, len(dataset)-len_train])

        # check sizes of images
    img_height = len(train_dataset[0][0][0])
    img_width = len(train_dataset[0][0][0][0])
    img_channels = len(train_dataset[0][0])


        # check output labels
    if ds == 'Other':
        num_labels = len(dataset.class_to_idx)
    elif ds == "CIFAR10" or ds == "CIFAR100":
        num_labels = (max(train_dataset.targets) - min(train_dataset.targets) + 1)
    else:
        num_labels = (max(train_dataset.targets) - min(train_dataset.targets) + 1).item()


        
    if IsLeNet == "LeNet":
        child_architecture = LeNet(img_height, img_width, num_labels, img_channels)
    elif IsLeNet == "EasyNet":
        child_architecture = EasyNet(img_height, img_width, num_labels, img_channels)
    elif IsLeNet == 'SimpleNet':
        child_architecture = SimpleNet(img_height, img_width, num_labels, img_channels)
    else:
        child_architecture = pickle.load(open(f'datasets/childnetwork', "rb"))

    return train_dataset, test_dataset, child_architecture 