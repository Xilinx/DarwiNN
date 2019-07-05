import time
import sys
import os
import torch
import torch.distributed as t_d
import torch.multiprocessing as mp
import copy
import numpy as np
import math
import argparse

class DarwiNNEnvironment(object):
    """Wrapper class for the environment setup API"""
    def __init__(self, cuda=True, seed=0):
        #initialize world (optimizer agnostic)
        self.number_nodes = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        self.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        print("Rank ",self.rank," (local: ",self.local_rank,") of ",self.number_nodes)
        #set environment variables for Gloo/NCCL
        os.environ['WORLD_SIZE'] = str(self.number_nodes)
        os.environ['RANK'] = str(self.rank) 
        #configure GPU environment if needed
        self.cuda = cuda
        if self.cuda:
            #pin each local rank to a GPU round-robin
            print("Using GPUs")
            backend = "nccl"
            print("Pin local rank ",self.local_rank," to GPU ",self.local_rank%torch.cuda.device_count()," of ",torch.cuda.device_count())
            torch.cuda.set_device(self.local_rank % torch.cuda.device_count())
            self.device = torch.device('cuda:'+str(torch.cuda.current_device()))
        else:
            print("Using CPUs")
            backend = "gloo"
            self.device = torch.device('cpu')
        #initialize Torch Distributed environment
        t_d.init_process_group(backend=backend, rank=self.rank, world_size=self.number_nodes) ##set group
        #configure local multiprocessing
        mp.set_start_method('spawn', force = True)
        torch.manual_seed(seed)
        #TODO: auto-tune
        #e.g. compute the maximum folding given the GPU/host memory, device, number of GPUs, and local ranks
        #TODO: selectable gpu/cpu collectives
        
    def rank(self):
        return self.rank
    
    def broadcast(self, x, src):
        t_d.broadcast(x,src=src)
    
    def gather(self, x, dst, dst_list):
        t_d.gather(tensor=x, dst=dst, gather_list=dst_list)
    
    def scatter(self):
        raise NotImplementedError
    
    def all_gather(self, x, dst_list):
        t_d.all_gather(tensor_list=dst_list, tensor=x)
    
    def all_reduce(self, x):
        t_d.all_reduce(x, op=t_d.reduce_op.SUM)

class DarwiNNOptimizer(object):
    """Abstract class for optimizer functions"""
    def __init__(self, environment, popsize=100, data_parallel=False):
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
            #all-gather fitness to dapt theta
            self.environment.all_gather(self.fitness_local,self.fitness_list)
            torch.cat(self.fitness_list, out=self.fitness_global)
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
    
class OpenAIESOptimizer(DarwiNNOptimizer):
    """Implements Open-AI ES optimizer"""
    def __init__(self, environment, model, criterion, optimizer, distribution="Gaussian", sampling="Antithetic", sigma=0.1, popsize=100, data_parallel=False):
        super(OpenAIESOptimizer,self).__init__(environment, popsize, data_parallel)
        self.optimizer = optimizer
        self.distribution = distribution
        self.sampling = sampling
        self.model_adapt = model
        self.model = copy.deepcopy(model)
        if self.environment.cuda:
            self.model_adapt.cuda()
            self.model.cuda()
        self.num_parameters = self.count_num_parameters()
        self.criterion = criterion
        self.sigma = sigma
        self.theta = torch.zeros((self.num_parameters), device=self.environment.device)
        self.update_theta()
        if (self.distribution == "Gaussian"):
            self.randfunc = torch.randn
        elif (self.distribution == "Uniform"):
            self.randfunc = torch.rand
        else:
            raise ValueError
        self.loss = 0

    def count_num_parameters(self):
        orig_params = []
        for param in self.model.parameters():
            p = param.data.cpu().numpy()
            orig_params.append(p.flatten())
        orig_params_flat = np.concatenate(orig_params)
        return len(orig_params_flat)

    def compute_centered_ranks(self,x):
        centered = torch.zeros(len(x), dtype = torch.float, device = self.environment.device)
        sort, ind = x.sort()
        for i in range(len(x)):
            centered[ind[i].data] = i
        centered = torch.div(centered, len(x) - 1)
        centered = centered - 0.5
        return centered

    def select(self):
        self.fitness_global = self.compute_centered_ranks(-self.fitness_global)

    def adapt(self):
        #regenerate noise
        epsilon = torch.zeros((self.popsize,self.num_parameters), device=self.environment.device)
        for i in range(self.popsize):
            epsilon[i] = self.gen_epsilon(i//self.folds,i%self.folds)
        #update model
        gradient = torch.mm(epsilon.t(), self.fitness_global.view(len(self.fitness_global), 1))
        self.update_grad(-gradient)
        self.optimizer.step()
        self.update_theta()
        #update local model from theta
        self.update_model(self.theta)
    
    def gen_noise(self):
        return self.randfunc(self.num_parameters, device=self.environment.device)*self.sigma
    
    def gen_epsilon(self, rank, fold_index):
        noise_id = self.generation*self.popsize + rank*self.folds + fold_index
        torch.manual_seed(noise_id)
        if (self.sampling == "Antithetic") and (fold_index >= int(self.folds/2)):
            noise_id = noise_id - int(self.folds/2)
            torch.manual_seed(noise_id)
            epsilon = -self.gen_noise()
        else:
            epsilon = self.gen_noise()
        return epsilon

    def mutate(self, fold_index):
        epsilon = self.gen_epsilon(self.environment.rank, fold_index)
        theta_noisy = self.theta + epsilon
        return theta_noisy
    
    def eval_theta(self, data, target):
        output = self.model_adapt(data)
        self.loss = self.criterion(output, target).item()
        return output
    
    def eval_fitness(self, data, target):
        #for each in local population, mutate then evaluate, resulting in a list of fitnesses
        for i in range(self.folds):
            self.update_model(self.mutate(i))
            output = self.model(data)
            self.fitness_local[i] = self.criterion(output, target).item()
        self.loss = torch.mean(self.fitness_local).item()
        
    def get_loss(self):
        return self.loss
        
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
