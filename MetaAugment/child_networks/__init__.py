from .lenet import *
from .bad_lenet import *

class LeNet(nn.Module):
    def __init__(self, img_height=28, img_width=28, num_labels=10, img_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(int((((img_height-4)/2-4)/2)*(((img_width-4)/2-4)/2)*16), 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_labels)
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


"""Define internal NN module that trains on the dataset"""
class EasyNet(nn.Module):
    def __init__(self, img_height=28, img_width=28, num_labels=10, img_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(img_height*img_width*img_channels, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, num_labels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = self.fc1(y)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        return y



"""Define internal NN module that trains on the dataset"""
class SimpleNet(nn.Module):
    def __init__(self, img_height=28, img_width=28, num_labels=10, img_channels=1):
        super().__init__()
        self.fc1 = nn.Linear(img_height*img_width*img_channels, num_labels)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = self.fc1(y)
        y = self.relu1(y)
        return y
