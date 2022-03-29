import torch
import numpy as np
from MetaAugment.main import *
import MetaAugment.child_networks as cn
import torchvision.transforms as transforms
from MetaAugment.autoaugment_learners.autoaugment import *

import torchvision.transforms.autoaugment as torchaa
from torchvision.transforms import functional as F, InterpolationMode

policies1 = [
            (("Invert", 0.8, None), ("Contrast", 0.2, 6)),
            (("Rotate", 0.7, 2), ("Invert", 0.8, None)),
            (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
            (("ShearY", 0.5, 8), ("Invert", 0.7, None)),
            (("AutoContrast", 0.5, None), ("Equalize", 0.9, None))
            ]

# We will use this augmentation_space temporarily. Later on we will need to 
# make sure we are able to add other image operations if the users want.
num_bins = 10
augmentation_space = [
            # (operation_name, do_we_need_to_specify_magnitude)
            ("ShearX", True),
            ("ShearY", True),
            ("TranslateX", True),
            ("TranslateY", True),
            ("Rotate", True),
            ("Brightness", True),
            ("Color", True),
            ("Contrast", True),
            ("Sharpness", True),
            ("Posterize", True),
            ("Solarize", True),
            ("AutoContrast", False),
            ("Equalize", False),
            ("Invert", False),
        ]

class randomsearch_learner:
    def __init__(self, sp_num=5):
        '''
        Args:
            spdim: number of subpolicies per policy
        '''
        self.sp_num = sp_num

        # op_num is the number of different operations
        # TODO: Allow op_num to be changed with the user's specifications 
        self.op_num = 14

    def generate_new_policy(self):
        '''
        Generate a new random policy in the form of
            [
            (("Invert", 0.8, None), ("Contrast", 0.2, 6)),
            (("Rotate", 0.7, 2), ("Invert", 0.8, None)),
            (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
            (("ShearY", 0.5, 8), ("Invert", 0.7, None)),
            ]
        '''


        def generate_new_discrete_subpolicy(op_num=14, p_bins=10, m_bins=10):
            '''
            generate a new random subpolicy in the form of a tensor of dimension:
                2* (op_num + 11 + 10)
            Where the 2 represents the 2 operations. 

            And for each operation, the first op_num dimensions are hard-maxed to specify which operation to use.
            The next 11 dimensions specify which 'probability' to choose.
                (0.0, 0.1, ..., 1.0)
            The next 10 dimensions specify which 'magnitude' to choose.
                (0, 1, ..., 9)
            '''
            op = np.random.randint(0, op_num)
            prob = np.random.randint(p_bins+1, op_num)
            mag = np.random.randint(m_bins, op_num)
            
            op_t= torch.zeros(op_num)
            op_t[op] = 1
            prob_t = torch.zeros(p_bins+1)
            mag_t = torch.zeros(m_bins)

            return torch.cat([op_t, prob_t, mag_t])
        

        def generate_new_continuous_subpolicy(op_num=14, p_bins=10, m_bins=10):
            '''
            Returns subpolicy_tensor, which is a tensor representation of a random subpolicy with
            dimension:
                2 * (op_num + 1 + 1)
            Where the 2 represents the 2 operations. 

            And for each operation, the first op_num dimensions are hard-maxed to specify which operation to use.
            The next 1 dimensions specify which 'probability' to choose.
                0 < x < 1
            The next 1 dimensions specify which 'magnitude' to choose.
                0 < x < 9
            
            Yes, the inequalities are strict, but we will round them up so that they go 
            to the nearest integers, so it will be possible for the magnitude to be 0 or 9
            '''
            def get_op_p_m():
                op = np.random.randint(0, op_num)
                prob = np.random.randint(p_bins+1, op_num)
                mag = np.random.randint(m_bins, op_num)
                
                op_p_m = torch.zeros(op_num + 2)
                op_p_m[op] = 1
                op_p_m[-2] = np.random.uniform()
                op_p_m[-1] = np.random.uniform() * 9

                return op_p_m
            
            return torch.cat([get_op_p_m, get_op_p_m])
        
        def translate_subpolicy_tensor(subpolicy_tensor, op_num=14,
                                        p_bins=10, m_bins=10,
                                        discrete_p_m=False):
            '''
            takes in a tensor representing a subpolicy and returns an actual subpolicy which
            is in the form of:
                (("Invert", 0.8, None), ("Contrast", 0.2, 6)))

            Args:
                subpolicy_tensor
                continuous_p_m (boolean): whether the subpolicy_tensor has continuous representations
                                        of probability and magnitude
            '''
            # if input subpolicy_tensor is continuous
            if not discrete_p_m:
                op_t = subpolicy_tensor[:op_num]
                prob_t = subpolicy_tensor[op_num:op_num+p_bins+1]
                mag_t = subpolicy_tensor[-m_bins:]
                raise NotImplementedError, "U can implement this if u want"
            
            # process continuous subpolicy_tensor
            op_t = subpolicy_tensor[:op_num]
            p = subpolicy_tensor[-2] # 0 < p < 1
            m = subpolicy_tensor[-1] # 0 < m < 9

            op_num = torch.argmax(op_t)
            p = round(p, 1) # round to nearest first decimal digit
            m = round(m) # round to nearest integer


            # take argmax


        new_policy = []
        for _ in range(self.sp_num):
            new_subpolicy = generate_new_continuous_subpolicy(self.op_num)


            new_policy.append(new_subpolicy)

        return new_policy

    def learn(self, train_dataset, test_dataset, child_network, toy_flag):
        '''
        Does the loop which is seen in Figure 1 in the AutoAugment paper.
        In other words, repeat:
            1. <generate a random policy>
            2. <see how good that policy is>
            3. <save how good the policy is in a list/dictionary>
        '''

        # test out 15 random policies
        for _ in range(15):
            policy = self.generate_new_policy()

            reward = self.test_autoaugment_policy(policy, child_network, train_dataset,
                                                test_dataset, toy_flag)

            self.controller.update(reward, policy)
        

    

    def test_autoaugment_policy(self, policy, child_network, train_dataset, test_dataset):
        '''
        Given a policy (using AutoAugment paper terminology), we train a child network
        with the policy and return the accuracy.
        '''
        # We need to define an object aa_transform which takes in the image and 
        # transforms it with the policy (specified in its .policies attribute)
        # in its forward pass
        aa_transform = AutoAugment()
        aa_transform.policies = policy
        train_transform = transforms.Compose([
                                                aa_transform,
                                                transforms.ToTensor()
                                            ])
        
        # We feed the transformation into the Dataset object
        train_dataset.transform = train_transform

        # create Dataloader objects out of the Dataset objects
        train_loader, test_loader = create_toy(train_dataset,
                                                test_dataset,
                                                batch_size=32,
                                                n_samples=0.01,
                                                seed=100)

        # train the child network with the dataloaders equipped with our specific policy
        accuracy = train_child_network(child_network, 
                                    train_loader, 
                                    test_loader, 
                                    sgd = optim.SGD(child_network.parameters(), lr=1e-1),
                                    cost = nn.CrossEntropyLoss(),
                                    max_epochs = 100, 
                                    early_stop_num = 10, 
                                    logging = False)
        return accuracy


if __name__=='__main__':

    # We can initialize the train_dataset with its transform as None.
    # Later on, we will change this object's transform attribute to the policy
    # that we want to test
    train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, download=False, 
                                transform=None)
    test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, download=False,
                                transform=torchvision.transforms.ToTensor())
    child_network = cn.lenet()

    
    rs_learner = randomsearch_learner()
    rs_learner.learn(train_dataset, test_dataset, child_network, toy_flag=True)
    print(rs_learner.best_five)