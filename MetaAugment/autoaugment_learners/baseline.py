import MetaAugment.child_networks as cn
from pprint import pprint
import torchvision.datasets as datasets
import torchvision
from MetaAugment.autoaugment_learners.aa_learner import aa_learner

train_dataset = datasets.MNIST(root='./MetaAugment/datasets/mnist/train',
                                train=True, download=True, transform=None)
test_dataset = datasets.MNIST(root='./MetaAugment/datasets/mnist/test', 
                        train=False, download=True, transform=torchvision.transforms.ToTensor())
child_network = cn.bad_lenet

aalearner = aa_learner(discrete_p_m=True)

# this policy is same as identity function, because probabaility and magnitude are both zero
null_policy = [(("Contrast", 0.0, 0.0), ("Contrast", 0.0, 0.0))]

aalearner.test_autoaugment_policy(null_policy, child_network(), train_dataset, test_dataset, 
                                toy_flag=True, logging=False)

pprint(aalearner.history)