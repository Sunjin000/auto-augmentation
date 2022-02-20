import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import torchvision.transforms.autoaugment as autoaugment
import random
import pygad
import pygad.torchga as torchga
import random
#import MetaAugment.AutoAugmentDemo.ops as ops # 

np.random.seed(0)
random.seed(0)


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class Learner(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 13)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)

        idx_ret = torch.argmax(y[:, 0:3].mean(dim = 0))
        p_ret = 0.1 * torch.argmax(y[:, 3:].mean(dim = 0))
        return (idx_ret, p_ret)


def train_model(transform_idx, p):
    """
    Takes in the specific transformation index and probability 
    """

    if transform_idx == 0:
        transform_train = torchvision.transforms.Compose(
           [
            torchvision.transforms.RandomVerticalFlip(p),
            torchvision.transforms.ToTensor(),
            ]
               )
    elif transform_idx == 1:
        transform_train = torchvision.transforms.Compose(
           [
            torchvision.transforms.RandomHorizontalFlip(p),
            torchvision.transforms.ToTensor(),
            ]
               )
    else:
        transform_train = torchvision.transforms.Compose(
           [
            torchvision.transforms.RandomGrayscale(p),
            torchvision.transforms.ToTensor(),
            ]
               )

    batch_size = 32
    n_samples = 0.05

    train_dataset = datasets.MNIST(root='./MetaAugment/train', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root='./MetaAugment/test', train=False, download=True, transform=torchvision.transforms.ToTensor())

    shuffled_train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)).tolist())
    indices_train = torch.arange(int(n_samples*len(train_dataset)))
    reduced_train_dataset = torch.utils.data.Subset(shuffled_train_dataset, indices_train)

    shuffled_test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)).tolist())
    indices_test = torch.arange(int(n_samples*len(test_dataset)))
    reduced_test_dataset = torch.utils.data.Subset(shuffled_test_dataset, indices_test)


    train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(reduced_test_dataset, batch_size=batch_size)

    model = LeNet()
    sgd = optim.SGD(model.parameters(), lr=1e-1)
    cost = nn.CrossEntropyLoss()
    epoch = 5

    for _epoch in range(epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = cost(predict_y, train_label.long())
            loss.backward()
            sgd.step()

        correct = 0
        _sum = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_ys = np.argmax(predict_y, axis=-1)
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        if _epoch % 1 == 0:
            print('Epoch: {} \t Accuracy: {:.2f}%'.format(_epoch, correct / _sum *100))
        #torch.save(model, f'mnist_{correct / _sum}.pkl')
    return correct / _sum *100



def fitness_func(solution, sol_idx):
    """
    Defines fitness function (accuracy of the model)
    """
    global train_loader, meta_rl_agent

    model_weights_dict = torchga.model_weights_as_dict(model=meta_rl_agent,
                                                       weights_vector=solution)
    # Use the current solution as the model parameters.
    meta_rl_agent.load_state_dict(model_weights_dict)
    for idx, (test_x, label_x) in enumerate(train_loader):
        trans_idx, p = meta_rl_agent(test_x)

    fit_val = train_model(trans_idx, p)

    return fit_val


def callback_generation(ga_instance):
    """
    Just prints stuff while running
    """
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def mutation_func(offspring, ga_instance):
    global sig, tau

    sig = sig * np.exp( tau * np.random.normal(0, 1))

    offspring = offspring + (sig * np.random.normal(0, 1, offspring.shape))

    return offspring


def crossover_func(parents, offspring_size, ga_instance):

    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_split_point = np.random.choice(range(offspring_size[0]))

        parent1[random_split_point:] = parent2[random_split_point:]

        offspring.append(parent1)

        idx += 1

    return np.array(offspring)

# ORGANISING DATA


train_dataset = datasets.MNIST(root='./MetaAugment/train', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = datasets.MNIST(root='./MetaAugment/test', train=False, download=True, transform=torchvision.transforms.ToTensor())
n_samples = 0.02
# shuffle and take first n_samples  %age of training dataset
shuffled_train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)).tolist())
indices_train = torch.arange(int(n_samples*len(train_dataset)))
reduced_train_dataset = torch.utils.data.Subset(shuffled_train_dataset, indices_train)

# shuffle and take first n_samples %age of test dataset
shuffled_test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset)).tolist())
indices_test = torch.arange(int(n_samples*len(test_dataset)))
reduced_test_dataset = torch.utils.data.Subset(shuffled_test_dataset, indices_test)

train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=60000)



# GENERATING THE GA INSTANCE

meta_rl_agent = Learner()
torch_ga = torchga.TorchGA(model=meta_rl_agent,
                           num_solutions=20)

# HYPERPARAMETER FOR THE GA 

num_generations = 10 # Number of generations.
num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights
tau = 1 / np.sqrt(initial_population[0].shape)
sig = 1


ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       parent_selection_type="rank",
                       fitness_func=fitness_func,
                       on_generation=callback_generation, 
                       mutation_type = mutation_func,
                       save_solutions = True)
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

best_solution_weights = torchga.model_weights_as_dict(model=meta_rl_agent,
                                                      weights_vector=solution)


ga_instance.plot_fitness()







# print("WITH p = 0")


# train_model(0,0)





# def meta_rl():
#     train_dataset = datasets.MNIST(root='./MetaAugment/train', train=True, download=True, transform=torchvision.transforms.ToTensor())
#     shuffled_train_dataset = torch.utils.data.Subset(train_dataset, torch.randperm(len(train_dataset)).tolist())
#     n_samples = 0.02
#     indices_train = torch.arange(int(n_samples*len(train_dataset)))
#     reduced_train_dataset = torch.utils.data.Subset(shuffled_train_dataset, indices_train)
#     train_loader = torch.utils.data.DataLoader(reduced_train_dataset, batch_size=60000)

#     for idx, (train_inputs, train_label) in enumerate(train_loader):

#         sample_shape = train_inputs.shape[1:]
#         num_classes = 13

#         input_layer = pygad.cnn.Input2D(input_shape=sample_shape)
#         conv_layer1 = pygad.cnn.Conv2D(num_filters=2,
#                                     kernel_size=3,
#                                     previous_layer=input_layer,
#                                     activation_function=None)
#         relu_layer1 = pygad.cnn.Sigmoid(previous_layer=conv_layer1)
#         average_pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2,
#                                                         previous_layer=relu_layer1,
#                                                         stride=2)

#         conv_layer2 = pygad.cnn.Conv2D(num_filters=3,
#                                     kernel_size=3,
#                                     previous_layer=average_pooling_layer,
#                                     activation_function=None)
#         relu_layer2 = pygad.cnn.ReLU(previous_layer=conv_layer2)
#         max_pooling_layer = pygad.cnn.MaxPooling2D(pool_size=2,
#                                                 previous_layer=relu_layer2,
#                                                 stride=2)

#         conv_layer3 = pygad.cnn.Conv2D(num_filters=1,
#                                     kernel_size=3,
#                                     previous_layer=max_pooling_layer,
#                                     activation_function=None)
#         relu_layer3 = pygad.cnn.ReLU(previous_layer=conv_layer3)
#         pooling_layer = pygad.cnn.AveragePooling2D(pool_size=2,
#                                                 previous_layer=relu_layer3,
#                                                 stride=2)

#         flatten_layer = pygad.cnn.Flatten(previous_layer=pooling_layer)
#         dense_layer1 = pygad.cnn.Dense(num_neurons=100,
#                                     previous_layer=flatten_layer,
#                                     activation_function="relu")

#         dense_layer2 = pygad.cnn.Dense(num_neurons=num_classes,
#                                     previous_layer=dense_layer1,
#                                     activation_function="sigmoid")

#         model = pygad.cnn.Model(last_layer=dense_layer2,
#                                 epochs=1,
#                                 learning_rate=0.01)