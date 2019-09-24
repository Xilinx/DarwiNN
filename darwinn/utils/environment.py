#  Copyright (c) 2019, Xilinx
#  All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#  
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#  
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#  
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

    #performs data synchronization between workers
    def synchronize(self, x, mode="NONE", lst=None):
        if mode == "NONE":
            pass
        elif mode == "AVERAGE":
            self.all_reduce(x)
            x /= self.number_nodes
        elif mode == "GATHER":
            if self.number_nodes > 1:
                self.all_gather(x,lst)
            else: #work-around for bug in Gloo for np=1
                pass
        else:
            raise Exception("Illegal synchronization mode")