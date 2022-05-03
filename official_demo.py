import autoaug
import autoaug.autoaugment_learners as aal

import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
### Defining our CNN Classifier
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
### Defining the training and validation datasets
import torchvision.datasets as datasets

train_dataset = datasets.MNIST(
                        root='./autoaug/datasets/mnist/train',
                        train=True,
                        download=True,
                        transform=None
                        )
val_dataset = datasets.MNIST(
                        root='./autoaug/datasets/mnist/test',
                        train=False,
                        download=True,
                        transform=torchvision.transforms.ToTensor()
                        )
### Defining the child network architecture
child_network_architecture = LeNet
### specifying parameters for the auto-augment learner
search_space_hyp = {
        'exclude_method': ['Invert', 'Solarize']
        }
child_network_hyp = {
        'learning_rate': 0.01,
        'early_stop_num': 5,
        'batch_size': 32,
        'toy_size': 0.0025
        }


learner = aal.RsLearner(
                        **search_space_hyp,
                        **child_network_hyp,
                        )
### Training the auto-augment learner
learner.learn(
        train_dataset=train_dataset,
        test_dataset=val_dataset,
        child_network_architecture=child_network_architecture,
        iterations = 9)
### Viewing the Results
print(learner.get_n_best_policies(3))

breakpoint()