import torch

import MetaAugment.child_networks as cn
from MetaAugment.autoaugment_learners.aa_learner import aa_learner
from MetaAugment.controller_networks.rnn_controller import RNNModel

from pprint import pprint
import pickle



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

    def __init__(self, sp_num=5, fun_num=14, p_bins=11, m_bins=10, discrete_p_m=True, alpha=0.2):
        '''
        Args:
            spdim: number of subpolicies per policy
            fun_num: number of image functions in our search space
            p_bins: number of bins we divide the interval [0,1] for probabilities
            m_bins: number of bins we divide the magnitude space

            alpha: Exploration parameter. The lower this value, the more exploration.
        '''
        super().__init__(sp_num, fun_num, p_bins, m_bins, discrete_p_m=True)
        self.alpha = alpha

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
        log_prob = 0

        # we need a random input to put in
        random_input = torch.zeros(self.rnn_output_size, requires_grad=False)

        # 2*self.sp_num because we need 2 operations for every subpolicy
        vectors = self.controller(input=random_input, time_steps=2*self.sp_num)

        # softmax the funcion vector, probability vector, and magnitude vector
        # of each timestep
        softmaxed_vectors = []
        for vector in vectors:
            fun_t, prob_t, mag_t = vector.split([self.fun_num, self.p_bins, self.m_bins])
            fun_t = self.softmax(fun_t * self.alpha)
            prob_t = self.softmax(prob_t * self.alpha)
            mag_t = self.softmax(mag_t * self.alpha)
            softmaxed_vector = torch.cat((fun_t, prob_t, mag_t))
            softmaxed_vectors.append(softmaxed_vector)
            
        new_policy = []

        for subpolicy_idx in range(self.sp_num):
            # the vector corresponding to the first operation of this subpolicy
            op1 = softmaxed_vectors[2*subpolicy_idx]
            # the vector corresponding to the second operation of this subpolicy
            op2 = softmaxed_vectors[2*subpolicy_idx+1]

            # translate both vectors
            op1, log_prob1 = self.translate_operation_tensor(op1, return_log_prob=True)
            op2, log_prob2 = self.translate_operation_tensor(op2, return_log_prob=True)
            
            new_policy.append((op1,op2))
            log_prob += (log_prob1+log_prob2)
        
        return new_policy, log_prob


    def learn(self, train_dataset, test_dataset, child_network_architecture, toy_flag, m=8):
        '''
        Does the loop which is seen in Figure 1 in the AutoAugment paper.
        In other words, repeat:
            1. <generate a random policy>
            2. <see how good that policy is>
            3. <save how good the policy is in a list/dictionary>
        '''
        # optimizer for training the GRU controller
        cont_optim = torch.optim.SGD(self.controller.parameters(), lr=1e-2)

        m = 8 # minibatch size
        b = 0.88 # b is the running exponential mean of the rewards, used for training stability
               # (see section 3.2 of https://arxiv.org/abs/1611.01578)

        for _ in range(1000):
            cont_optim.zero_grad()

            # obj(objective) is $ \sum_{k=1}^m (reward_k-b) \sum_{t=1}^T log(P(a_t|a_{(t-1):1};\theta_c))$,
            # which is used in PPO
            obj = 0

            # sum up the rewards within a minibatch in order to update the running mean, 'b'
            mb_rewards_sum = 0

            for k in range(m):
                # log_prob is $\sum_{t=1}^T log(P(a_t|a_{(t-1):1};\theta_c))$, used in PPO
                policy, log_prob = self.generate_new_policy()

                pprint(policy)
                child_network = child_network_architecture()
                reward = self.test_autoaugment_policy(policy, child_network, train_dataset,
                                                    test_dataset, toy_flag)
                mb_rewards_sum += reward

                # log
                self.history.append((policy, reward))

                # gradient accumulation
                obj += (reward-b)*log_prob
            
            # update running mean of rewards
            b = 0.7*b + 0.3*(mb_rewards_sum/m)

            (-obj).backward() # We put a minus because we want to maximize the objective, not 
                              # minimize it.
            cont_optim.step()

            # save the history every 1 epochs as a pickle
            with open('gru_logs.pkl', 'wb') as file:
                pickle.dump(self.history, file)
            with open('gru_learner.pkl', 'wb') as file:
                pickle.dump(self, file)
             



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
