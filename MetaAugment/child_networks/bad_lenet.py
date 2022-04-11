import torch.nn as nn


# class Bad_LeNet(nn.Module):
#     # 1. I reduced the channel sizes of the convolutional layers
#     # 2. I reduced the number of fully ocnnected layers from 3 to 2
#     # 
#     # no. of weights: 25*2 + 25*2*4 + 16*4*10 = 250+640 = 890
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 2, 5)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(2, 4, 5)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(16*4,  10)
#         self.relu3 = nn.ReLU()


#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.relu1(y)
#         y = self.pool1(y)
#         y = self.conv2(y)
#         y = self.relu2(y)
#         y = self.pool2(y)
#         y = y.view(y.shape[0], -1)
#         y = self.fc1(y)
#         y = self.relu3(y)
#         return y

class Bad_LeNet(nn.Module):
    # 1. I reduced the channel sizes of the convolutional layers
    # 2. I reduced the number of fully connected layers from 3 to 2
    # 
    # no. of weights: 25*2 + 25*2*3 + 4*3*10 = 50+150+120 = 320
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(2, 3, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(4*3,  10)
        self.relu3 = nn.ReLU()
        
        # self.fc2 = nn.Linear(20, 14)
        # self.relu4 = nn.ReLU()
        # self.fc3 = nn.Linear(14, 10)
        # self.relu5 = nn.ReLU()


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
        # y = self.fc2(y)
        # y = self.relu4(y)
        # y = self.fc3(y)
        # y = self.relu5(y)
        return y


def bad_lenet():
    model = Bad_LeNet()
    return model