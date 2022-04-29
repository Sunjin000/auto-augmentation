import autoaug.autoaugment_learners as aal
import autoaug.child_networks as cn
import torch
import torchvision
import torchvision.datasets as datasets

import random


def test__translate_operation_tensor():
    """
    See if AaLearner class's _translate_operation_tensor works
    by feeding many (valid) inputs in it.

    We make a lot of (fun_num+p_bins_m_bins,) size tensors, softmax 
    them, and feed them through the _translate_operation_tensor method
    to see if it doesn't break
    """

 
    
    # discrete_p_m=True

    for i in range(2000):

        softmax = torch.nn.Softmax(dim=0)

        fun_num=14
        p_bins = random.randint(2, 15)
        m_bins = random.randint(2, 15)
        
        agent = aal.AaLearner(
                sp_num=5,
                p_bins=p_bins,
                m_bins=m_bins,
                discrete_p_m=True
                )

        alpha = i/1000
        vector = torch.rand(fun_num+p_bins+m_bins)
        fun_t, prob_t, mag_t = vector.split([fun_num, p_bins, m_bins])
        fun_t = softmax(fun_t * alpha)
        prob_t = softmax(prob_t * alpha)
        mag_t = softmax(mag_t * alpha)
        softmaxed_vector = torch.cat((fun_t, prob_t, mag_t))

        agent._translate_operation_tensor(softmaxed_vector)
    

    # discrete_p_m=False
    softmax = torch.nn.Softmax(dim=0)
    sigmoid = torch.nn.Sigmoid()
    for i in range(2000):
        

        fun_num = 14
        p_bins = random.randint(1, 15)
        m_bins = random.randint(1, 15)

        agent = aal.AaLearner(
                sp_num=5,
                p_bins=p_bins,
                m_bins=m_bins,
                discrete_p_m=False
                )

        alpha = i/1000
        vector = torch.rand(fun_num+2)
        fun_t, prob_t, mag_t = vector.split([fun_num, 1, 1])
        fun_t = softmax(fun_t * alpha)
        prob_t = sigmoid(prob_t)
        mag_t = sigmoid(mag_t) * (m_bins-1)

        softmaxed_vector = torch.cat((fun_t, prob_t, mag_t))

        agent._translate_operation_tensor(softmaxed_vector)


def test__test_autoaugment_policy():
    agent = aal.AaLearner(
                sp_num=5,
                p_bins=11,
                m_bins=10,
                discrete_p_m=True,
                toy_size=0.002,
                max_epochs=20,
                early_stop_num=10
                )
    

    policy = [
            (("Invert", 0.8, None), ("Contrast", 0.2, 6)),
            (("Rotate", 0.7, 2), ("Invert", 0.8, None)),
            (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
            (("ShearY", 0.5, 8), ("Invert", 0.7, None)),
            (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
            (("ShearY", 0.8, 4), ("Rotate", 0.5, 6)),
            (("TranslateY", 0.7, 4), ("TranslateX", 0.8, 6)),
            (("Rotate", 0.5, 3), ("ShearY", 0.8, 5)),
            (("ShearX", 0.5, 6), ("TranslateY", 0.7, 3)),
            (("Rotate", 0.5, 3), ("TranslateX", 0.5, 5))
            ]
    child_network_architecture = cn.SimpleNet
    train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=None)
    test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                            train=False, download=True,
                            transform=torchvision.transforms.ToTensor())

    acc = agent._test_autoaugment_policy(
                                        policy,
                                        child_network_architecture,
                                        train_dataset,
                                        test_dataset,
                                        logging=False
                                        )
    
    assert isinstance(acc, float)


def test_exclude_method():
    """
    we want to see if the exclude_methods
    parameter is working properly in aa_learners 
    """
    
    exclude_method = [
                    'ShearX', 
                    'Color', 
                    'Brightness', 
                    'Contrast'
                    ]
    agent = aal.GruLearner(
        exclude_method=exclude_method
    )
    for _ in range(200):
        new_pol, _ = agent._generate_new_policy()
        print(new_pol)
        for (op1, op2) in new_pol:
            image_function_1 = op1[0]
            image_function_2 = op2[0]
            assert image_function_1 not in exclude_method
            assert image_function_2 not in exclude_method
    
    agent = aal.RsLearner(
        exclude_method=exclude_method
    )
    for _ in range(200):
        new_pol= agent._generate_new_policy()
        print(new_pol)
        for (op1, op2) in new_pol:
            image_function_1 = op1[0]
            image_function_2 = op2[0]
            assert image_function_1 not in exclude_method
            assert image_function_2 not in exclude_method
    

def test_get_mega_policy():

    agent = aal.RsLearner(
                sp_num=5,
                p_bins=11,
                m_bins=10,
                discrete_p_m=True,
                toy_size=0.002,
                max_epochs=20,
                early_stop_num=10
                )

    child_network_architecture = cn.SimpleNet
    train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=None)
    test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                            train=False, download=True,
                            transform=torchvision.transforms.ToTensor())

    agent.learn(train_dataset, test_dataset, child_network_architecture, 10)
    mega_pol = agent.get_mega_policy(number_policies=30)
    mega_pol = agent.get_mega_policy(number_policies=3)
    mega_pol = agent.get_mega_policy(number_policies=1)
    print("megapol: ", mega_pol)


if __name__=='__main__':
    test_get_mega_policy()