import args
import time
import sys
import os
import torch
import torch.distributed as t_d
import multiprocessing as mp
import copy
import numpy as np
import math
import argparse

class DarwiNNEnvironment(object):
    """Wrapper class for the environment setup API"""
    def __init__(self, cuda=True, seed=0):
        #initialize world (optimizer agnostic)
        #initialize MPI environment
        self.number_nodes = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        self.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        print("Rank ",self.rank," (local: ",self.local_rank,") of ",self.number_nodes)
        t_d.init_process_group(backend='mpi', rank=self.rank, world_size=self.number_nodes) ##set group
        #configure local multiprocessing
        mp.set_start_method('spawn', force = True)
        #configure GPU environment if needed
        torch.manual_seed(seed)
        print("Pin local rank ",self.local_rank," to GPU ",self.local_rank%torch.cuda.device_count()," of ",torch.cuda.device_count())
        self.cuda = cuda
        if cuda:
            #pin each local rank to a GPU round-robin
            torch.cuda.set_device(self.local_rank % torch.cuda.device_count())
            torch.cuda.manual_seed(seed)
        #TODO: auto-tune stuff could go in here
        
    def rank(self):
        return self.rank
    
    #TODO: collective ops, implemented with pytorch distributed backend
    def broadcast(self, x, src):
        t_d.broadcast(x,src=src)
    
    def gather(self, x, dst, dst_list):
        t_d.gather(tensor=x, dst=dst, gather_list=dst_list)
    
    def scatter(self):
        raise NotImplementedError
    
    def allgather(self):
        raise NotImplementedError
    
    def allreduce(self):
        raise NotImplementedError

class DarwiNNOptimizer(object):
    """Abstract class for optimizer functions"""
    
    #adapts a model according to results of a fitness evaluation
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
    def __init__(self, environment, model, criterion, optimizer, distribution="Gaussian", sampling="Antithetic", sigma=0.1, population=100, device = torch.device('cpu')):
        super(DarwiNNOptimizer, self).__init__()
        self.environment = environment
        self.optimizer = optimizer
        self.distribution = distribution
        self.sampling = sampling
        self.model = model
        self.criterion = criterion
        self.device = device
        self.population = (population // environment.number_nodes) * environment.number_nodes #round down requested population to something divisible by number of nodes
        self.sigma = sigma
        self.folds = self.population / environment.number_nodes # local population size TODO: in case of Antitethic check if local popsize divisible by 2
        self.loss_adapt_list = [torch.zeros((folding,), device=self.environment.device) for i in range(self.environment.number_nodes)]
        self.loss_adapt = torch.zeros((population,), device=self.environment.device)
        self.fitness = torch.zeros((folding,), device=self.environment.device)
        if (self.distribution == "Gaussian"):
            self.randfunc = torch.randn
        elif (self.distribution == "Uniform"):
            self.randfunc = torch.rand
        else:
            raise ValueError
        self.loss = 0
        self.generation = 1

    def count_num_parameters(self):
        orig_params = []
        for param in self.model.parameters():
            p = param.data.cpu().numpy()
            orig_params.append(p.flatten())
        orig_params_flat = np.concatenate(orig_params)
        return len(orig_params_flat) 
    
    def update_model(self, flat_param):
        idx = 0
        i = 0
        for param in self.model.parameters():
            flattened_dim = param.numel()
            temp = flat_param[idx:idx+flattened_dim]
            temp = temp.view_as(param)
            param.data = temp.data

    def compute_centered_ranks(self,x):
        centered = torch.zeros(len(x), dtype = torch.float, device = self.device)
        sort, ind = x.sort()
        for i in range(len(x)):
            centered[ind[i].data] = i
        centered = torch.div(centered, len(x) - 1)
        centered = centered - 0.5
        return centered

    def adapt(self, num_parameters):
        #regenerate noise
        epsilon = []
        for i in range(self.environment.number_nodes):
            epsilon += gen_epsilon(i)
        #gather fitness to node 0 to adapt theta
        self.environment.gather(self.fitness,0,self.loss_adapt_list)
        torch.cat(self.loss_adapt_list, out=self.loss_adapt)
        #update model on rank 0
        if self.rank == 0:
            self.loss_adapt = self.compute_centered_ranks(-self.loss_adapt)
            gradient = torch.mm(epsilon.t(), self.loss_adapt.view(len(self.loss_adapt), 1))
            step = self.optimizer.step(-gradient)
            self.theta = self.theta + step.view(num_parameters)
        #broadcast
        self.environment.broadcast(self.theta,0)
        #update local model from (broadcast) theta
        update_model()
        self.generation += 1
    
    def gen_noise(self):
        return self.randfunc(self.num_parameters, device=self.device)*self.sigma
    
    def gen_epsilon(self,rank):
        for fold_index in range(self.folds):
            #generate PRNG seed from generation, rank and fold index
            noise_id = self.generation*self.population + rank*self.folds + fold_index
            torch.manual_seed(noise_id)
            if (self.sampling == "Antithetic"):
                if (fold_index < int(self.folds/2)):
                    epsilon[fold_index] = self.gen_noise()
                else:
                    epsilon[fold_index] = -epsilon_minion[fold_index - int(self.folds/2)]
            else:
                epsilon[fold_index] = self.gen_noise()
        return epsilon

    def mutate(self):
        epsilon = gen_epsilon(self.environment.rank)
        theta_noisy = self.theta + epsilon
        return theta_noisy
    
    def eval_theta(data, target):
        output = self.model(data, target)
        self.loss = criterion(output, target).item()
        return output
    
    def eval_fitness(data, target):
        #for each in local population, mutate then evaluate, resulting in a list of fitnesses
        loss = []
        for i in range(self.folds):
            mutated_theta = mutate()
            output = mutated_theta(data)
            loss += [self.criterion(output, target)]
        
    def get_loss():
        return self.loss
        
    """Updates the NN model from the value of Theta"""
    def update_model():
        idx = 0
        i = 0
        for param in self.model.parameters():
            flattened_dim = param.numel()
            temp = self.theta[idx:idx+flattened_dim]
            temp = temp.view_as(param)
            param.data = temp.data
