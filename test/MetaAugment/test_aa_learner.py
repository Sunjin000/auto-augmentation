import MetaAugment.autoaugment_learners as aal
import MetaAugment.child_networks as cn
import torch
import torchvision
import torchvision.datasets as datasets

import random


def test_translate_operation_tensor():
    """
    See if aa_learner class's translate_operation_tensor works
    by feeding many (valid) inputs in it.

    We make a lot of (fun_num+p_bins_m_bins,) size tensors, softmax 
    them, and feed them through the translate_operation_tensor method
    to see if it doesn't break
    """

 
    
    # discrete_p_m=True

    for i in range(2000):

        softmax = torch.nn.Softmax(dim=0)

        fun_num=14
        p_bins = random.randint(2, 15)
        m_bins = random.randint(2, 15)
        
        agent = aal.aa_learner(
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

        agent.translate_operation_tensor(softmaxed_vector)
    

    # discrete_p_m=False
    softmax = torch.nn.Softmax(dim=0)
    sigmoid = torch.nn.Sigmoid()
    for i in range(2000):
        

        fun_num = 14
        p_bins = random.randint(1, 15)
        m_bins = random.randint(1, 15)

        agent = aal.aa_learner(
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

        agent.translate_operation_tensor(softmaxed_vector)


def test_test_autoaugment_policy():
    agent = aal.aa_learner(
                sp_num=5,
                p_bins=11,
                m_bins=10,
                discrete_p_m=True,
                toy_size=0.004,
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

    acc = agent.test_autoaugment_policy(
                                        policy,
                                        child_network_architecture,
                                        train_dataset,
                                        test_dataset,
                                        logging=False
                                        )
    
    assert isinstance(acc, float)
    