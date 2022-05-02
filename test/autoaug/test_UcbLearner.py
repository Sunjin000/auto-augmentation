import autoaug.autoaugment_learners as aal
import autoaug.child_networks as cn
import torchvision
import torchvision.datasets as datasets
from pprint import pprint

def test_ucb_learner():
    child_network_architecture = cn.SimpleNet
    train_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/train',
                            train=True, download=True, transform=None)
    test_dataset = datasets.FashionMNIST(root='./datasets/fashionmnist/test', 
                            train=False, download=True,
                            transform=torchvision.transforms.ToTensor())


    learner = aal.UcbLearner(
        # parameters that define the search space
                num_sub_policies=5,
                p_bins=11,
                m_bins=10,
                # hyperparameters for when training the child_network
                batch_size=8,
                toy_size=0.001,
                learning_rate=1e-1,
                max_epochs=float('inf'),
                early_stop_num=30,
                # UcbLearner specific hyperparameter
                num_policies=3
    )
    pprint(learner.policies)
    assert len(learner.policies)==len(learner.avg_accs), \
                (len(learner.policies), (len(learner.avg_accs)))

    # learn on the 3 policies we generated
    learner.learn(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        child_network_architecture=child_network_architecture,
        iterations=5
        )
    
    # let's say we want to explore more policies:
    # we generate more new policies
    learner.make_more_policies(n=4)

    # and let's explore how good those are as well
    learner.learn(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        child_network_architecture=child_network_architecture,
        iterations=7
        )

    print(learner.get_mega_policy(number_policies=50))
    print(learner.get_mega_policy(number_policies=3))

if __name__=="__main__":
    test_ucb_learner()
