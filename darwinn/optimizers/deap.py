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
from darwinn import DarwiNNOptimizer
from deap import tools

class DEAPeaGenerateUpdateOptimizer(DarwiNNOptimizer):
    """Implements a generic black-box optimizer where objective, mutation, and adaptation functions are defined externally"""
    def __init__(self, environment, population, dimension, objective_fn, update_fn, generate_fn, halloffame=None, stats=None, verbose=True):
        super(DEAPeaGenerateUpdateOptimizer,self).__init__(environment, population)
        self.objective_fn = objective_fn
        self.generate_fn = generate_fn
        self.update_fn = update_fn
        self.halloffame = halloffame
        self.population = []
        self.stats = stats
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + (self.stats.fields if stats else [])
        self.verbose = verbose

    def eval_fitness(self):
        #for each in local population, mutate then evaluate, resulting in a list of fitnesses
        for i in range(self.folds):
            self.fitness_local[i] = self.objective_fn(self.population[self.folds*self.environment.local_rank+i])[0]

    def step(self):
        self.population = self.generate_fn()
        # Evaluate the individuals
        self.eval_fitness()
        #all-gather fitness to dapt theta
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
