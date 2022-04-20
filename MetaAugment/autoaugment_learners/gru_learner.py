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
    """
    An AutoAugment learner with a GRU controller 

    The original AutoAugment paper(http://arxiv.org/abs/1805.09501) 
    uses a LSTM controller updated via Proximal Policy Optimization.
    (See Section 3 of AutoAugment paper)

    The GRU has been shown to be as powerful of a sequential neural
    network as the LSTM whilst training and testing much faster
    (https://arxiv.org/abs/1412.3555), which is why we substituted
    the LSTM for the GRU.
    """

    def __init__(self,
                # parameters that define the search space
                sp_num=5,
                fun_num=14,
                p_bins=11,
                m_bins=10,
                discrete_p_m=False,
                # hyperparameters for when training the child_network
                batch_size=8,
                toy_flag=False,
                toy_size=0.1,
                learning_rate=1e-1,
                max_epochs=float('inf'),
                early_stop_num=20,
                # GRU-specific attributes that aren't in all other aa_learners's
                alpha=0.2,
                cont_mb_size=8):
        """
        Args:
            alpha (float, optional): Exploration parameter. It is multiplied to 
                    operation tensors before they're softmaxed. 
                    The lower this value, the more smoothed the output
                    of the softmaxed will be, hence more exploration.
                    Defaults to 0.2.
            cont_mb_size (int, optional): Controller Minibatch Size. How many
                    policies do we test in order to calculate the 
                    PPO(proximal policy update) gradient to update
                    the controller. Defaults to 
        """
        if discrete_p_m==True:
            print('Warning: Incompatible discrete_p_m=True input into gru_learner. \
                discrete_p_m=False will be used')
        
        super().__init__(sp_num, 
                fun_num, 
                p_bins, 
                m_bins, 
                discrete_p_m=True, 
                batch_size=batch_size, 
                toy_flag=toy_flag, 
                toy_size=toy_size, 
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                early_stop_num=early_stop_num,)

        # GRU-specific attributes that aren't in general aa_learner's
        self.alpha = alpha
        self.cont_mb_size = cont_mb_size

        # CONTROLLER (GRU NETWORK) SETTINGS
        self.controller = RNNModel(mode='GRU', output_size=self.op_tensor_length, 
                                    num_layers=2, bias=True)
        self.cont_optim = torch.optim.SGD(self.controller.parameters(), lr=1e-2)

        self.softmax = torch.nn.Softmax(dim=0)


    def generate_new_policy(self):
        """
        The GRU controller pops out a new policy.

        At each time step, the GRU outputs a 
        (fun_num + p_bins + m_bins, ) dimensional tensor which 
        contains information regarding which 'image function' to use,
        which value of 'probability(prob)' and 'magnitude(mag)' to use.

        We run the GRU for 2*self.sp_num timesteps to obtain 2*self.sp_num
        of such tensors.

        We then softmax the parts of the tensor which represents the
        choice of function, prob, and mag seperately, so that the
        resulting tensor's values sums up to 3.

        Then we input each tensor into self.translate_operation_tensor
        with parameter (return_log_prob=True), which outputs a tuple
        in the form of ('img_function_name', prob, mag) and a float
        representing the log probability that we chose the chosen 
        func, prob and mag. 

        We add up the log probabilities of each operation.

        We turn the operations into a list of 5 tuples such as:
            [
            (("Invert", 0.8, None), ("Contrast", 0.2, 6)),
            (("Rotate", 0.7, 2), ("Invert", 0.8, None)),
            (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
            (("ShearY", 0.5, 8), ("Invert", 0.7, None)),
            ]
        This list can then be input into an AutoAugment object
        as is done in self.learn()
        
        We return a tuple of the list and the sum of the log probs
        """

        log_prob = 0

        # we need a random input to put in
        random_input = torch.zeros(self.op_tensor_length, requires_grad=False)

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


    def learn(self, 
            train_dataset, 
            test_dataset, 
            child_network_architecture, 
            iterations=15,):

        b = 0.5 # b is the running exponential mean of the rewards, used for training stability
               # (see section 3.2 of https://arxiv.org/abs/1611.01578)

        for _ in range(iterations):
            self.cont_optim.zero_grad()

            # obj(objective) is $ \sum_{k=1}^m (reward_k-b) \sum_{t=1}^T log(P(a_t|a_{(t-1):1};\theta_c))$,
            # which is used in PPO
            obj = 0

            # sum up the rewards within a minibatch in order to update the running mean, 'b'
            mb_rewards_sum = 0

            for k in range(self.cont_mb_size):
                # log_prob is $\sum_{t=1}^T log(P(a_t|a_{(t-1):1};\theta_c))$, used in PPO
                policy, log_prob = self.generate_new_policy()

                pprint(policy)
                reward = self.test_autoaugment_policy(policy,
                                                    child_network_architecture, 
                                                    train_dataset,
                                                    test_dataset)
                mb_rewards_sum += reward

                # log
                self.history.append((policy, reward))

                # gradient accumulation
                obj += (reward-b)*log_prob
            
            # update running mean of rewards
            b = 0.7*b + 0.3*(mb_rewards_sum/self.cont_mb_size)

            (-obj).backward() # We put a minus because we want to maximize the objective, not 
                              # minimize it.
            self.cont_optim.step()

            # # save the history every 1 epochs as a pickle
            # with open('gru_logs.pkl', 'wb') as file:
            #     pickle.dump(self.history, file)
            # with open('gru_learner.pkl', 'wb') as file:
            #     pickle.dump(self, file)
             



if __name__=='__main__':

    # We can initialize the train_dataset with its transform as None.
    # Later on, we will change this object's transform attribute to the policy
    # that we want to test
    import torchvision.datasets as datasets
    import torchvision
    torch.manual_seed(0)

    # train_dataset = datasets.MNIST(root='./datasets/mnist/train',
    #                                 train=True, download=True, transform=None)
    # test_dataset = datasets.MNIST(root='./datasets/mnist/test', 
    #                         train=False, download=True, 
    #                         transform=torchvision.transforms.ToTensor())
    train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=None)
    test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                            train=False, download=True,
                            transform=torchvision.transforms.ToTensor())
    child_network_architecture = cn.lenet
    # child_network_architecture = cn.lenet()

    agent = gru_learner(
                        sp_num=7,
                        toy_flag=True,
                        toy_size=0.01,
                        batch_size=4,
                        learning_rate=0.05,
                        max_epochs=float('inf'),
                        early_stop_num=35,
                        )
    agent.learn(train_dataset,
                test_dataset,
                child_network_architecture=child_network_architecture,
                iterations=3)
    
    pprint(agent.history)
