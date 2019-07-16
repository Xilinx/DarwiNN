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
        t_d.all_reduce(x, op=t_d.ReduceOp.SUM)

