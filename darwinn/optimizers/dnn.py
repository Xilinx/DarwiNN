import torch
import torch.distributed as t_d
import torch.multiprocessing as mp
import copy
import numpy as np
import math
from darwinn.utils.fitness import compute_centered_ranks

class DarwiNNOptimizer(object):
    """Abstract class for optimizer functions"""
    def __init__(self, environment, model, criterion, popsize=100, data_parallel=False):
        #Disable Autograd
        torch.autograd.set_grad_enabled(False)
        #set environment
        self.environment = environment
        #set number of population-parallel working nodes
        self.data_parallel = data_parallel
        if self.data_parallel:
            self.nodes = 1
        else:
            self.nodes = self.environment.number_nodes
        #round down requested population to something divisible by number of ranks
        self.popsize = (popsize // self.nodes) * self.nodes
        #evenly divide population between ranks
        self.folds = self.popsize // self.nodes
        print("NE Optimizer parameters: population=",self.popsize,", folds=",self.folds)
        #define data structures to hold fitness values
        self.fitness_list = [torch.zeros((self.folds,), device=self.environment.device) for i in range(self.nodes)]
        self.fitness_global = torch.zeros((self.popsize,), device=self.environment.device)
        self.fitness_local = torch.zeros((self.folds,), device=self.environment.device)
        #initialize model and theta
        self.model_adapt = model
        self.model = copy.deepcopy(model)
        if self.environment.cuda:
            self.model.cuda()
            self.model_adapt.cuda()
        self.num_parameters = self.count_num_parameters()
        self.criterion = criterion
        self.theta = torch.zeros((self.num_parameters), device=self.environment.device)
        self.update_theta()
        self.loss = 0
        #initialize generation
        self.generation = 1
    
    #performs selection and adaption according to results of a fitness evaluation
    def step(self):
        if self.data_parallel:
            for i in range(self.folds):
                #all-reduce all of the local fitnesses
                self.environment.all_reduce(self.fitness_local[i])
            #average local fitness sum to obtain global fitness 
            self.fitness_global = self.fitness_local / self.environment.number_nodes
        else:
            if self.environment.number_nodes > 1:
                #all-gather fitness to dapt theta
                self.environment.all_gather(self.fitness_local,self.fitness_list)
                torch.cat(self.fitness_list, out=self.fitness_global)
            else: #work-around for bug in Gloo for np=1
                self.fitness_global = self.fitness_local
        self.select()
        self.adapt()
        self.generation += 1
    
    #select by fitness and prepare for next generation
    def select(self):
        raise NotImplementedError
    
    #select by fitness and prepare for next generation
    def adapt(self):
        raise NotImplementedError
    
    #transforms a model for the next generation
    def mutate(self):
        raise NotImplementedError
    
    def eval_fitness(self, data, target):
        raise NotImplementedError
    
    def eval_theta(self, data, target):
        raise NotImplementedError
    
    def get_loss(self):
        raise NotImplementedError

    def count_num_parameters(self):
        orig_params = []
        for param in self.model.parameters():
            p = param.data.cpu().numpy()
            orig_params.append(p.flatten())
        orig_params_flat = np.concatenate(orig_params)
        return len(orig_params_flat)

    """Updates the NN model from the value of Theta"""
    def update_model(self, theta):
        idx = 0
        for param in self.model.parameters():
            flattened_dim = param.numel()
            temp = theta[idx:idx+flattened_dim]
            temp = temp.view_as(param)
            param.data = temp.data
            idx += flattened_dim

    """Updates the NN model gradients"""
    def update_grad(self, grad):
        idx = 0
        for param in self.model_adapt.parameters():
            flattened_dim = param.numel()
            temp = grad[idx:idx+flattened_dim]
            temp = temp.view_as(param.data)
            param.grad = temp.data
            idx += flattened_dim

    """Updates Theta from the NN model"""
    def update_theta(self):
        idx = 0
        for param in self.model_adapt.parameters():
            flattened_dim = param.numel()
            self.theta[idx:idx+flattened_dim] = param.data.flatten()
            idx += flattened_dim
            
    def eval_theta(self, data, target):
        output = self.model_adapt(data)
        self.loss = self.criterion(output, target).item()
        return output
        
    def get_loss(self):
        return self.loss
    
class OpenAIESOptimizer(DarwiNNOptimizer):
    """Implements Open-AI ES optimizer"""
    def __init__(self, environment, model, criterion, optimizer, distribution="Gaussian", sampling="Antithetic", sigma=0.1, popsize=100, data_parallel=False):
        super(OpenAIESOptimizer,self).__init__(environment, model, criterion, popsize, data_parallel)
        self.optimizer = optimizer
        self.distribution = distribution
        self.sampling = sampling
        self.sigma = sigma
        self.epsilon = torch.zeros((self.popsize,self.num_parameters), device=self.environment.device)
        if not data_parallel:
            self.fold_offset = self.environment.rank*self.folds
        else:
            self.fold_offset = 0
        if (self.distribution == "Gaussian"):
            self.randfunc = torch.randn
        elif (self.distribution == "Uniform"):
            self.randfunc = torch.rand
        else:
            raise ValueError

    def select(self):
        self.fitness_global = compute_centered_ranks(-self.fitness_global, device=self.environment.device)

    def adapt(self):
        gradient = torch.mm(self.epsilon.t(), self.fitness_global.view(len(self.fitness_global), 1))
        self.update_grad(-gradient)
        self.optimizer.step()
        self.update_theta()
        #update local model from theta
        self.update_model(self.theta)
    
    def gen_epsilon(self):
        if (self.sampling == "Antithetic"):
            half_epsilon = self.randfunc((self.popsize//2,self.num_parameters), device=self.environment.device)*self.sigma
            opposite_epsilon = half_epsilon*-1.0
            self.epsilon = torch.cat((half_epsilon,opposite_epsilon),0)
        else:
            self.epsilon = self.randfunc((self.popsize,self.num_parameters), device=self.environment.device)*self.sigma

    def mutate(self, fold_index):
        theta_noisy = self.theta + self.epsilon[fold_index+self.fold_offset]
        return theta_noisy
    
    def eval_fitness(self, data, target):
        self.gen_epsilon()
        for i in range(self.folds):
            self.update_model(self.mutate(i))
            output = self.model(data)
            self.fitness_local[i] = self.criterion(output, target).item()
        self.loss = torch.mean(self.fitness_local).item()

class GAOptimizer(DarwiNNOptimizer):
    """Implements a simple Genetic Algorithm optimizer"""
    def __init__(self, environment, model, criterion, sigma=0.1, popsize=100, elite_ratio=0.1, mutation_probability=0.01, data_parallel=False):
        super(GAOptimizer,self).__init__(environment, model, criterion, popsize, data_parallel)
        self.fold_offset = self.environment.rank*self.folds
        self.num_elites = int(self.popsize*elite_ratio)
        self.elites = torch.zeros((self.num_elites,self.num_parameters), device=self.environment.device)
        self.population = torch.zeros((self.popsize,self.num_parameters), device=self.environment.device)
        self.sigma = sigma

    def select(self):
        #sort by fitness
        fitness_sorted, ind = self.fitness_global.sort()
        #select elites from population using indices of top fitnesses
        self.elites = torch.index_select(self.population,0,ind[:self.num_elites])   
                
    def mutate(self):
        #elites are inherited
        for i in range(self.num_elites):
            self.population[i] = self.elites[i]
        #rest of population is generated
        for i in range(self.num_elites,self.popsize):
            idx1 = torch.randint(0,self.num_elites,(1,),device=self.environment.device)
            idx2 = torch.randint(0,self.num_elites,(1,),device=self.environment.device)
            parent_1 = self.elites[idx1.item()]
            parent_2 = self.elites[idx2.item()]
            parent1_select = torch.randint(0,2,(self.num_parameters,), dtype=torch.float, device=self.environment.device)
            parent2_select = (parent1_select - 1.0) * -1.0
            #crossover
            self.population[i] = self.elites[idx1.item()] * parent1_select
            self.population[i] += parent_2 * parent2_select
            #mutation
            self.population[i] += torch.randn((self.num_parameters,), device=self.environment.device)*self.sigma
            
    def adapt(self):
        pass

    def eval_fitness(self, data, target):
        self.mutate()
        for i in range(self.folds):
            self.update_model(self.population[self.fold_offset+i])
            output = self.model(data)
            self.fitness_local[i] = self.criterion(output, target).item()
        self.loss = torch.mean(self.fitness_local).item()
        