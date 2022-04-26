import torch
torch.manual_seed(0)
import torch.nn as nn
import pygad
import pygad.torchga as torchga
import copy
import torch

from MetaAugment.autoaugment_learners.aa_learner import aa_learner


class evo_learner(aa_learner):

    def __init__(self, 
                # search space settings
                sp_num=5,
                p_bins=10, 
                m_bins=10, 
                discrete_p_m=False,
                exclude_method=[],
                # child network settings
                learning_rate=1e-1, 
                max_epochs=float('inf'),
                early_stop_num=20,
                batch_size=8,
                toy_size=1,
                # evolutionary learner specific settings
                num_solutions=5,
                num_parents_mating=3,
                controller=None
                ):

        super().__init__(
                    sp_num=sp_num, 
                    p_bins=p_bins, 
                    m_bins=m_bins, 
                    discrete_p_m=discrete_p_m, 
                    batch_size=batch_size, 
                    toy_size=toy_size, 
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    early_stop_num=early_stop_num,
                    exclude_method=exclude_method
                    )

        self.num_solutions = num_solutions
        self.controller = controller
        self.torch_ga = torchga.TorchGA(model=self.controller, num_solutions=num_solutions)
        self.num_parents_mating = num_parents_mating
        self.initial_population = self.torch_ga.population_weights
        self.p_bins = p_bins 
        self.sub_num_pol = sp_num
        self.m_bins = m_bins
        self.policy_dict = {}
        self.policy_result = []


        assert num_solutions > num_parents_mating, 'Number of solutions must be larger than the number of parents mating!'


    def get_full_policy(self, x):
        """
        Generates the full policy (self.num_sub_pol subpolicies). Network architecture requires
        output size 5 * 2 * (self.fun_num + self.p_bins + self.m_bins)

        Parameters 
        -----------
        x -> PyTorch tensor
            Input data for network 

        Returns
        ----------
        full_policy -> [((String, float, float), (String, float, float)), ...)
            Full policy consisting of tuples of subpolicies. Each subpolicy consisting of
            two transformations, with a probability and magnitude float for each
        """
        section = self.fun_num + self.p_bins + self.m_bins
        y = self.controller.forward(x)
        full_policy = []
        for pol in range(self.sub_num_pol):
            int_pol = []
            for _ in range(2):
                idx_ret = torch.argmax(y[:, (pol * section):(pol*section) + self.fun_num].mean(dim = 0))

                trans, need_mag = self.augmentation_space[idx_ret]

                if self.p_bins == 1:
                    p_ret = min(1, max(0, (y[:, (pol * section)+self.fun_num:(pol*section)+self.fun_num+self.p_bins].mean(dim = 0).item())))
                    # p_ret = torch.sigmoid(y[:, (pol * section)+self.fun_num:(pol*section)+self.fun_num+self.p_bins].mean(dim = 0))
                else:
                    p_ret = torch.argmax(y[:, (pol * section)+self.fun_num:(pol*section)+self.fun_num+self.p_bins].mean(dim = 0).item()) * 0.1


                if need_mag:
                    # print("original mag", y[:, (pol * section)+self.fun_num+self.p_bins:((pol+1)*section)].mean(dim = 0))
                    if self.m_bins == 1:
                        mag = min(9, max(0, (y[:, (pol * section)+self.fun_num+self.p_bins:((pol+1)*section)].mean(dim = 0).item())))
                    else:
                        mag = torch.argmax(y[:, (pol * section)+self.fun_num+self.p_bins:((pol+1)*section)].mean(dim = 0).item())
                    mag = int(mag)
                else:
                    mag = None
                int_pol.append((trans, p_ret, mag))

            full_policy.append(tuple(int_pol))

        return full_policy

    
    def get_single_policy_cov(self, x, alpha = 0.5):
        """
        Selects policy using population and covariance matrices. For this method 
        we require p_bins = 1, num_sub_pol = 1, m_bins = 1. 

        Parameters
        ------------
        x -> PyTorch Tensor
            Input data for the AutoAugment network 

        alpha -> Float
            Proportion for covariance and population matrices 

        Returns
        -----------
        Subpolicy -> [(String, float, float), (String, float, float)]
            Subpolicy consisting of two tuples of policies, each with a string associated 
            to a transformation, a float for a probability, and a float for a magnittude
        """
        section = self.fun_num + self.p_bins + self.m_bins

        y = self.controller.forward(x)

        y_1 = torch.softmax(y[:,:self.fun_num], dim = 1) 
        y[:,:self.fun_num] = y_1
        y_2 = torch.softmax(y[:,section:section+self.fun_num], dim = 1)
        y[:,section:section+self.fun_num] = y_2
        concat = torch.cat((y_1, y_2), dim = 1)

        cov_mat = torch.cov(concat.T)
        cov_mat = cov_mat[:self.fun_num, self.fun_num:]
        shape_store = cov_mat.shape

        counter, prob1, prob2, mag1, mag2 = (0, 0, 0, 0, 0)


        prob_mat = torch.zeros(shape_store)
        for idx in range(y.shape[0]):
            prob_mat[torch.argmax(y_1[idx])][torch.argmax(y_2[idx])] += 1
        prob_mat = prob_mat / torch.sum(prob_mat)

        cov_mat = (alpha * cov_mat) + ((1 - alpha)*prob_mat)

        cov_mat = torch.reshape(cov_mat, (1, -1)).squeeze()
        max_idx = torch.argmax(cov_mat)
        val = (max_idx//shape_store[0])
        max_idx = (val, max_idx - (val * shape_store[0]))


        if not self.augmentation_space[max_idx[0]][1]:
            mag1 = None
        if not self.augmentation_space[max_idx[1]][1]:
            mag2 = None
    
        for idx in range(y.shape[0]):
            if (torch.argmax(y_1[idx]) == max_idx[0]) and (torch.argmax(y_2[idx]) == max_idx[1]):
                prob1 += torch.sigmoid(y[idx, self.fun_num]).item()
                prob2 += torch.sigmoid(y[idx, section+self.fun_num]).item()
                if mag1 is not None:
                    # mag1 += min(max(0, (y[idx, self.auto_aug_agent.fun_num+1]).item()), 8)
                    mag1 += 10 * torch.sigmoid(y[idx, self.fun_num+1]).item()
                if mag2 is not None:
                    # mag2 += min(max(0, y[idx, section+self.auto_aug_agent.fun_num+1].item()), 8)
                    mag2 += 10 * torch.sigmoid(y[idx, self.fun_num+1]).item()

                counter += 1

        prob1 = round(prob1/counter, 1) if counter != 0 else 0
        prob2 = round(prob2/counter, 1) if counter != 0 else 0
        if mag1 is not None:
            mag1 = int(mag1/counter)
        if mag2 is not None:
            mag2 = int(mag2/counter)  

        
        return [((self.augmentation_space[max_idx[0]][0], prob1, mag1), (self.augmentation_space[max_idx[1]][0], prob2, mag2))]


    def learn(self, train_dataset, test_dataset, child_network_architecture, iterations = 15, return_weights = False):
        """
        Runs the GA instance and returns the model weights as a dictionary

        Parameters
        ------------
        return_weights -> Bool
            Determines if the weight of the GA network should be returned 
        
        Returns
        ------------
        If return_weights:
            Network weights -> Dictionary
        
        Else:
            Solution -> Best GA instance solution

            Solution fitness -> Float

            Solution_idx -> Int
        """
        self.num_generations = iterations
        self.history_best = [0 for i in range(iterations+1)]
        print("itations: ", iterations)

        self.history_avg = [0 for i in range(iterations+1)]
        self.gen_count = 0
        self.best_model = 0

        self.set_up_instance(train_dataset, test_dataset, child_network_architecture)
        print("train_dataset: ", train_dataset)

        self.ga_instance.run()
        self.history_avg = [x / self.num_solutions for x in self.history_avg]
        print("-----------------------------------------------------------------------------------------------------")

        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        if return_weights:
            return torchga.model_weights_as_dict(model=self.controller, weights_vector=solution)
        else:
            return solution, solution_fitness, solution_idx


    def in_pol_dict(self, new_policy):
        new_policy = new_policy[0]
        trans1, trans2 = new_policy[0][0], new_policy[1][0]
        new_set = {new_policy[0][1], new_policy[0][2], new_policy[1][1], new_policy[1][2]}
        if trans1 in self.policy_dict:
            if trans2 in self.policy_dict[trans1]:
                for test_pol in self.policy_dict[trans1][trans2]:
                    if new_set == test_pol:
                        return True
                self.policy_dict[trans1][trans2].append(new_set)
                return False 
            else:
                self.policy_dict[trans1][trans2] = [new_set]
        if trans2 in self.policy_dict:
            if trans1 in self.policy_dict[trans2]:
                for test_pol in self.policy_dict[trans2][trans1]:
                    if new_set == test_pol:
                        return True
                self.policy_dict[trans2][trans1].append(new_set)
                return False 
            else:
                self.policy_dict[trans2][trans1] = [new_set]


    def set_up_instance(self, train_dataset, test_dataset, child_network_architecture):
        """
        Initialises GA instance, as well as fitness and on_generation functions
        
        """

        def fitness_func(solution, sol_idx):
            """
            Defines the fitness function for the parent selection

            Parameters
            --------------
            solution -> GA solution instance (parsed automatically)

            sol_idx -> GA solution index (parsed automatically)

            Returns 
            --------------
            fit_val -> float            
            """

            model_weights_dict = torchga.model_weights_as_dict(model=self.controller,
                                                            weights_vector=solution)

            self.controller.load_state_dict(model_weights_dict)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

            for idx, (test_x, label_x) in enumerate(self.train_loader):
                # if self.sp_num == 1:
                full_policy = self.get_single_policy_cov(test_x)


                # else:                      
                # full_policy = self.get_full_policy(test_x)
                while self.in_pol_dict(full_policy):
                    full_policy = self.get_single_policy_cov(test_x)[0]


            fit_val = self.test_autoaugment_policy(full_policy,child_network_architecture,train_dataset,test_dataset) #) /
                      #  + self.test_autoaugment_policy(full_policy, train_dataset, test_dataset)) / 2

            self.policy_result.append([full_policy, fit_val])

            if len(self.policy_result) > self.sp_num:
                self.policy_result = sorted(self.policy_result, key=lambda x: x[1], reverse=True)
                self.policy_result = self.policy_result[:self.sp_num]
                print("Appended policy: ", self.policy_result)


            if fit_val > self.history_best[self.gen_count]:
                print("Best policy: ", full_policy)
                self.history_best[self.gen_count] = fit_val 
                self.best_model = model_weights_dict
            
            self.history_avg[self.gen_count] += fit_val
            

            return fit_val

        def on_generation(ga_instance):
            """
            Prints information of generational fitness

            Parameters 
            -------------
            ga_instance -> GA instance

            Returns
            -------------
            None
            """
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            self.gen_count += 1
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
            return


        self.ga_instance = pygad.GA(num_generations=self.num_generations, 
            num_parents_mating=self.num_parents_mating, 
            initial_population=self.initial_population,
            mutation_percent_genes = 0.1,
            fitness_func=fitness_func,
            on_generation = on_generation)

