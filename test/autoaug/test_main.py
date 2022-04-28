import torch
import torchvision
import torchvision.datasets as datasets

import autoaug.autoaugment_learners as aal
import autoaug.child_networks as cn
import autoaug.main as main

def test_create_toy():
    train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=None)
    test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                            train=False, download=True,
                            transform=torchvision.transforms.ToTensor())
    for _ in range(20):
        train_loader, test_loader = main.create_toy(train_dataset, test_dataset,
                            batch_size=32, n_samples=1)
        p = torch.rand_like(torch.tensor([.0]))
        train_loader, test_loader = main.create_toy(train_dataset, test_dataset,
                            batch_size=32, n_samples=p)
    
    
    train_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                            train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10(root='./datasets/cifar10/train',
                            train=False, download=True, 
                            transform=torchvision.transforms.ToTensor())
    for _ in range(20):
        train_loader, test_loader = main.create_toy(train_dataset, test_dataset,
                            batch_size=32, n_samples=1)
        p = torch.rand_like(torch.tensor([.0]))
        train_loader, test_loader = main.create_toy(train_dataset, test_dataset,
                            batch_size=32, n_samples=p)


def test_train_cn():
    train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True,
                            transform=torchvision.transforms.ToTensor())
    test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                            train=False, download=True,
                            transform=torchvision.transforms.ToTensor())
    cn_architecture = cn.Bad_LeNet
    model = cn_architecture()

    train_loader, test_loader = main.create_toy(train_dataset, test_dataset, 
                            batch_size=32, n_samples=0.01)

    main.train_child_network(
                            model, 
                            train_loader, 
                            test_loader,
                            sgd=torch.optim.SGD(model.parameters(),lr=0.1),
                            cost=torch.nn.CrossEntropyLoss(),
                            early_stop_flag=True
                            )
    
    main.train_child_network(
                            model, 
                            train_loader, 
                            test_loader,
                            sgd=torch.optim.SGD(model.parameters(),lr=0.1),
                            cost=torch.nn.CrossEntropyLoss(),
                            early_stop_flag=False
                            )