import math
import torch
import numpy as np
from enum import Enum

class NoiseMode(Enum):
    FULL = 1    # manipulate entire noise matrix
    SLICE_V = 2 # manipulate vertical slices of the matrix
    SLICE_H = 3 # manipulate horizontal slices of the matrix
    TILE = 4    # manipulate tiles created by applying both horizontal and vertical slicing

class NoiseGenerator(object):
    """Implements noise generation functions"""
    def __init__(self, popsize, nparams, device, nodes, rank, distribution="Gaussian", sampling="Antithetic", mutate_mode=NoiseMode.FULL, update_mode=NoiseMode.FULL):
        self.device = device
        self.nodes = nodes
        self.rank = rank
        self.distribution = distribution
        self.sampling = sampling
        self.popsize = popsize
        self.nparams = nparams
        #record readback modes for update and mutate phases
        self.mutate_mode = mutate_mode
        self.update_mode = update_mode
        #TODO: assert that popsize and nparams are divisible by the number of nodes (to get integer number of chunks in any direction)
        #set up buffers, indices, etc
        self.setup_buffers()
        # set RNG function according to requested distribution
        if (self.distribution == "Gaussian"):
            self.randfunc = torch.randn
        elif (self.distribution == "Uniform"):
            self.randfunc = torch.rand
        else:
            raise ValueError
        self.generation = 0
        #TODO: JIT/sequential generation of blocks to minimize memory utilization

    def setup_buffers(self):
        vslice_width_nominal = math.ceil(self.nparams/self.nodes)
        hslice_height_nominal = math.ceil(self.popsize/self.nodes)
        vslice_width_actual = min(vslice_width_nominal,self.nparams-self.rank*vslice_width_nominal)
        hslice_height_actual = min(hslice_height_nominal,self.popsize-self.rank*hslice_height_nominal)
        #assert that requested modes aren't TILE
        if (self.mutate_mode == self.update_mode):
            #allocate a buffer for the entire noise matrix
            if(self.mutate_mode == NoiseMode.FULL):
                self.noise = torch.empty((self.popsize,self.nparams),device=self.device)
            #allocate a buffer for a slice of the noise matrix
            elif (self.mutate_mode == NoiseMode.SLICE_H):
                self.noise = torch.empty((hslice_height_actual,self.nparams),device=self.device)
            else:
                self.noise = torch.empty((self.popsize,vslice_width_actual),device=self.device)
            self.mutate_noise = self.noise
            self.update_noise = self.noise
            self.mutate_blocks = (self.noise,)
            self.update_blocks = (self.noise,)
            #create boolean arrays indicating whether the blocks have been generated (avoid duplicating work)
            self.update_block_generated = np.zeros(1,dtype=bool)
            self.mutate_block_generated = self.update_block_generated[0:1]
            if (self.mutate_mode == NoiseMode.FULL):
                self.num_blocks = 1
                self.mutate_block_indices = np.array([0],dtype=np.int)
                self.update_block_indices = np.array([0],dtype=np.int)
            else:
                self.num_blocks = self.nodes
                self.mutate_block_indices = np.array([self.rank],dtype=np.int)
                self.update_block_indices = np.array([self.rank],dtype=np.int)
        elif (self.mutate_mode == NoiseMode.FULL or self.update_mode == NoiseMode.FULL):
            #allocate a buffer for the entire noise matrix
            self.noise = torch.empty((self.popsize,self.nparams),device=self.device)
            self.num_blocks = self.nodes
            #determine which is the non-full mode
            slice_mode = self.mutate_mode if self.update_mode==NoiseMode.FULL else self.update_mode
            #chunk the noise in the appropriate direction
            self.mutate_blocks = torch.chunk(self.noise,self.nodes,dim=(0 if slice_mode == NoiseMode.SLICE_H else 1))
            self.update_blocks = self.mutate_blocks
            #point mutation and update noise pointers to appropriate buffer
            self.mutate_noise = self.noise
            self.update_noise = self.noise
            self.mutate_block_indices = np.arange(self.num_blocks,dtype=np.int)
            self.update_block_indices = np.arange(self.num_blocks,dtype=np.int)
            self.num_blocks = self.nodes
            if(self.mutate_mode != NoiseMode.FULL):
                self.mutate_noise = self.update_blocks[self.rank]
                self.mutate_blocks = (self.update_blocks[self.rank],)
                self.mutate_block_indices = np.array([self.rank],dtype=np.int)
                self.update_block_generated = np.zeros(len(self.update_block_indices),dtype=bool)
                self.mutate_block_generated = self.update_block_generated[self.rank:self.rank+1]
            elif(self.update_mode != NoiseMode.FULL):
                self.update_noise = self.mutate_blocks[self.rank]
                self.update_blocks = (self.mutate_blocks[self.rank],)
                self.update_block_indices = np.array([self.rank],dtype=np.int)
                self.mutate_block_generated = np.zeros(len(self.mutate_block_indices),dtype=bool)
                self.update_block_generated = self.mutate_block_generated[self.rank:self.rank+1]
        elif (self.mutate_mode == NoiseMode.SLICE_H and self.update_mode == NoiseMode.SLICE_V):
            #we need to allocate separate buffers for the update and mutate noise, because they don't fully overlap
            #the get_noise function has to do the proper assembly
            self.mutate_noise = torch.empty((hslice_height_actual,self.nparams),device=self.device)
            self.update_noise = torch.empty((self.popsize,vslice_width_actual),device=self.device)
            self.num_blocks = self.nodes*self.nodes
            self.mutate_blocks = torch.chunk(self.mutate_noise,self.nodes,dim=1)
            self.update_blocks = torch.chunk(self.update_noise,self.nodes,dim=0)
            self.mutate_block_indices = np.arange(self.nodes,dtype=np.int)+self.rank*self.nodes
            self.update_block_indices = np.arange(self.nodes,dtype=np.int)*self.nodes+self.rank
            self.mutate_block_generated = np.zeros(len(self.mutate_block_indices),dtype=bool)
            self.update_block_generated = np.zeros(len(self.update_block_indices),dtype=bool)
        else:
            raise Exception("Unsupported mode combination")

    def generate_block(self,i,x,seed):
        s = seed+self.num_blocks*self.generation
        torch.manual_seed(s)
        if (self.sampling == "Antithetic"):
            #TODO assert that block height is even
            half_noise = torch.chunk(x,2,dim=0)
            self.randfunc(half_noise[0].shape,out=half_noise[0])
            half_noise[1].copy_(half_noise[0])
            half_noise[1].mul_(-1.0)
        else:
            self.randfunc(x.shape, out=x)

    def generate_noise(self,indices,blocks,flags):    
        for i in range(len(indices)):
            if not flags[i]:
                self.generate_block(i,blocks[i],indices[i])
                flags[i]=True

    def generate_update_noise(self): 
        self.generate_noise(self.update_block_indices,self.update_blocks,self.update_block_generated)
        return self.update_noise
    
    def generate_mutate_noise(self):    
        self.generate_noise(self.mutate_block_indices,self.mutate_blocks,self.mutate_block_generated)
        return self.mutate_noise
    
    def step(self):
        #advance seeds; no seed must ever be used twice
        self.generation += 1
        #reset flags
        for i in range(len(self.mutate_block_indices)):
            self.mutate_block_generated[i] = False
        for i in range(len(self.update_block_indices)):
            self.update_block_generated[i] = False