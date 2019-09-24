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
from deap import tools

class DEAPeaGenerateUpdateOptimizer():
    """Implements a generic black-box optimizer where objective, mutation, and adaptation functions are defined externally"""
    def __init__(self, environment, popsize, dimension, objective_fn, update_fn, generate_fn, halloffame=None, stats=None, verbose=True):
        self.environment = environment
        self.popsize = (popsize // self.environment.number_nodes) * self.environment.number_nodes
        #evenly divide population between ranks
        self.folds = self.popsize // self.environment.number_nodes
        self.fitness_list = [torch.zeros((self.folds,), device=self.environment.device) for i in range(self.environment.number_nodes)]
        self.fitness_global = torch.zeros((self.popsize,), device=self.environment.device)
        self.fitness_local = torch.zeros((self.folds,), device=self.environment.device)
        self.objective_fn = objective_fn
        self.generate_fn = generate_fn
        self.update_fn = update_fn
        self.halloffame = halloffame
        self.population = []
        self.stats = stats
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + (self.stats.fields if stats else [])
        self.verbose = verbose
        self.generation = 1

    def eval_fitness(self):
        #for each in local population, mutate then evaluate, resulting in a list of fitnesses
        for i in range(self.folds):
            self.fitness_local[i] = self.objective_fn(self.population[self.folds*self.environment.local_rank+i])[0]

    def step(self):
        self.population = self.generate_fn()
        # Evaluate the individuals
        self.eval_fitness()
        #all-gather fitness to dapt theta
        if self.environment.number_nodes == 1:
            self.fitness_global = self.fitness_local
        else:
            self.environment.all_gather(self.fitness_local,self.fitness_list)
            torch.cat(self.fitness_list, out=self.fitness_global)
        #write fitness values into individuals
        for individual, fitness_value in zip(self.population, self.fitness_global):
            individual.fitness.values = (fitness_value.item(),)

        #update DEAP hall of fame
        if self.halloffame is not None and self.environment.rank == 0:
            self.halloffame.update(self.population)

        #update 
        self.update_fn(self.population)

        #log statistics
        if self.environment.rank == 0:
            record = self.stats.compile(self.population) if self.stats is not None else {}
            self.logbook.record(gen=self.generation, nevals=len(self.population), **record)
            if self.verbose:
                print(self.logbook.stream)

        self.generation += 1
