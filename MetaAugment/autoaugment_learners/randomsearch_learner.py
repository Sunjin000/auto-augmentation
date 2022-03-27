from MetaAugment.main import *
import MetaAugment.child_networks as cn
import torchvision.transforms as transforms
from MetaAugment.autoaugment_learners.autoaugment import *

import torchvision.transforms.autoaugment as torchaa
from torchvision.transforms import functional as F, InterpolationMode

    
class randomsearch_learner:
    def __init__(self):
        pass

    def learn(self, train_dataset, test_dataset, child_network, res, toy_flag):
        '''
        Does the loop which is seen in Figure 1 in the AutoAugment paper.

        'res' stands for resolution of the discretisation of the search space. It could be
        a tuple, with first entry regarding probability, second regarding magnitude
        '''
        good_policy_found = False

        while not good_policy_found:
            policy = self.controller.pop_policy()

            train_loader, test_loader = create_toy(train_dataset, test_dataset,
                                                    batch_size=32, n_samples=0.005)

            reward = train_child_network(child_network, train_loader, test_loader, sgd, cost, epoch)

            self.controller.update(reward, policy)
        
        return good_policy

    def test_autoaugment_policy(policies):
        aa_transform = AutoAugment()
        aa_transform.policies = policies

        train_transform = transforms.Compose([
                                                aa_transform,
                                                transforms.ToTensor()
                                            ])


        train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, download=False, 
                                    transform=train_transform)
        test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, download=False,
                                    transform=torchvision.transforms.ToTensor())

        # create toy dataset from above uploaded data
        train_loader, test_loader = create_toy(train_dataset, test_dataset, batch_size, 0.01)

        child_network = cn.lenet()
        sgd = optim.SGD(child_network.parameters(), lr=1e-1)

        best_acc = train_child_network(child_network, train_loader, test_loader, sgd, cost, max_epochs=100)

        train_dataset




if __name__=='__main__':


    batch_size = 32
    n_samples = 0.005
    cost = nn.CrossEntropyLoss()

    policies1 = [
            (("Invert", 0.8, None), ("Contrast", 0.2, 6)),
            (("Rotate", 0.7, 2), ("Invert", 0.8, None)),
            (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
            (("ShearY", 0.5, 8), ("Invert", 0.7, None)),
            (("AutoContrast", 0.5, None), ("Equalize", 0.9, None))
            ]

    # The one that i hand crafted. You'll see that this one usually reaches a much
    # higher poerformance
    policies2 = [
            (("ShearY", 0.8, 4), ("Rotate", 0.5, 6)),
            (("TranslateY", 0.7, 4), ("TranslateX", 0.8, 6)),
            (("Rotate", 0.5, 3), ("ShearY", 0.8, 5)),
            (("ShearX", 0.5, 6), ("TranslateY", 0.7, 3)),
            (("Rotate", 0.5, 3), ("TranslateX", 0.5, 5))
            ]


    learner = RandomSearch_Learner()

