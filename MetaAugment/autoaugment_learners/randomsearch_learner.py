from MetaAugment.autoaugment_learners.autoaugment import *
import torchvision.transforms.autoaugment as torchaa

    
class RandomSearch_Learner:
    def __init__(self):
        pass

    def learn(self, train_dataset, test_dataset, child_network, res, toy_flag):
        '''
        Does what is seen in Figure 1 in the AutoAugment paper.

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




def randomsearch_learner():
    model = RandomSearch_Learner()
    return model


if __name__ == '__main__':
    learner = RandomSearch_Learner()