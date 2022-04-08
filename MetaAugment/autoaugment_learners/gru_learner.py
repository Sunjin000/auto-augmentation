import torch

import MetaAugment.child_networks as cn
from MetaAugment.autoaugment_learners.aa_learner import aa_learner
from MetaAugment.controller_networks.rnn_controller import RNNModel

from pprint import pprint



# We will use this augmentation_space temporarily. Later on we will need to 
# make sure we are able to add other image functions if the users want.
augmentation_space = [
            # (function_name, do_we_need_to_specify_magnitude)
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


class gru_learner(aa_learner):
    # Uses a GRU controller which is updated via Proximal Polixy Optimization
    # It is the same model use in
    # http://arxiv.org/abs/1805.09501
    # and
    # http://arxiv.org/abs/1611.01578

    def __init__(self, sp_num=5, fun_num=14, p_bins=11, m_bins=10, discrete_p_m=True):
        '''
        Args:
            spdim: number of subpolicies per policy
            fun_num: number of image functions in our search space
            p_bins: number of bins we divide the interval [0,1] for probabilities
            m_bins: number of bins we divide the magnitude space
        '''
        super().__init__(sp_num, fun_num, p_bins, m_bins, discrete_p_m=True)

        self.rnn_output_size = fun_num+p_bins+m_bins
        self.controller = RNNModel(mode='GRU', output_size=self.rnn_output_size, 
                                    num_layers=1, bias=True)
        self.softmax = torch.nn.Softmax(dim=0)


    def generate_new_policy(self):
        '''
        We run the GRU for 10 timesteps to obtain 10 operations.
        At each time step, it outputs a (fun_num + p_bins + m_bins) dimensional vector

        And then for each operation, we put it through self. 
        Generate a new policy in the form of
            [
            (("Invert", 0.8, None), ("Contrast", 0.2, 6)),
            (("Rotate", 0.7, 2), ("Invert", 0.8, None)),
            (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
            (("ShearY", 0.5, 8), ("Invert", 0.7, None)),
            ]
        '''
        # we need a random input to put in
        random_input = torch.rand(self.rnn_output_size, requires_grad=False)

        # 2*self.sp_num because we need 2 operations for every subpolicy
        vectors = self.controller(input=random_input, time_steps=2*self.sp_num)

        # softmax the funcion vector, probability vector, and magnitude vector
        # of each timestep
        softmaxed_vectors = []
        for vector in vectors:
            print(vector)
            fun_t, prob_t, mag_t = vector.split([self.fun_num, self.p_bins, self.m_bins])
            fun_t = self.softmax(fun_t)
            prob_t = self.softmax(prob_t)
            mag_t = self.softmax(mag_t)
            softmaxed_vector = torch.cat((fun_t, prob_t, mag_t))
            softmaxed_vectors.append(softmaxed_vector)
            
        print(softmaxed_vectors)
        new_policy = []

        for subpolicy_idx in range(self.sp_num):
            # the vector corresponding to the first operation of this subpolicy
            op1 = softmaxed_vectors[2*subpolicy_idx]
            # the vector corresponding to the second operation of this subpolicy
            op2 = softmaxed_vectors[2*subpolicy_idx+1]

            # translate both vectors
            op1 = self.translate_operation_tensor(op1)
            op2 = self.translate_operation_tensor(op2)
            
            print('new subpol:', (op1, op2))
            new_policy.append((op1,op2))
        
        return new_policy


    def learn(self, train_dataset, test_dataset, child_network_architecture, toy_flag):
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

            pprint(policy)
            child_network = child_network_architecture()
            reward = self.test_autoaugment_policy(policy, child_network, train_dataset,
                                                test_dataset, toy_flag)

            self.history.append((policy, reward))


if __name__=='__main__':

    # We can initialize the train_dataset with its transform as None.
    # Later on, we will change this object's transform attribute to the policy
    # that we want to test
    import torchvision.datasets as datasets
    import torchvision
    torch.manual_seed(0)

    train_dataset = datasets.MNIST(root='./datasets/mnist/train', train=True, download=True, 
                                transform=None)
    test_dataset = datasets.MNIST(root='./datasets/mnist/test', train=False, download=True,
                                transform=torchvision.transforms.ToTensor())
    child_network = cn.lenet

    
    learner = gru_learner(discrete_p_m=False)
    newpol = learner.generate_new_policy()
    learner.learn(train_dataset, test_dataset, child_network, toy_flag=True)
    pprint(learner.history)