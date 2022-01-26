import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import MetaAugment.AutoAugmentDemo.ops as ops # 

print(torch.__version__)


# code from https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/train.py
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


# code from https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/train.py
batch_size = 256
train_dataset = datasets.MNIST(root='./MetaAugment/train', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = datasets.MNIST(root='./MetaAugment/test', train=False, download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
model = LeNet()
sgd = optim.SGD(model.parameters(), lr=1e-1)
cost = nn.CrossEntropyLoss()
epoch = 100

for _epoch in range(epoch):
    model.train()
    for idx, (train_x, train_label) in enumerate(train_loader):
        label_np = np.zeros((train_label.shape[0], 10))
        sgd.zero_grad()
        predict_y = model(train_x.float())
        loss = cost(predict_y, train_label.long())
        if idx % 10 == 0:
            print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
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

    print(f'accuracy: {correct / _sum}')
    torch.save(model, f'models/mnist_{correct / _sum}.pkl')



# We use the root parameter to define where to save the data.
# The train parameter is set to true because we are initializing the MNIST training dataset.
# The download parameter is set to true because we want to download it if it’s not already present in our data folder.
# The transform parameter is set to none because we don’t want to apply any image manipulation transforms at this time. 
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
