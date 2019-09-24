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

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import benchmarks
from deap import cma
from deap import algorithms
import argparse
from darwinn.optimizers.deap import DEAPeaGenerateUpdateOptimizer
from darwinn.utils.environment import DarwiNNEnvironment

if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='DarwiNN distributed DEAP Example (Rastrigin+CMA-ES)')
    parser.add_argument('--dimension', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--generations', type=int, default=100, metavar='N',
                        help='how many generations to run')
    parser.add_argument('--popsize', type=int, default=1000, metavar='N',
                        help='population size (default: 1000)')
    args = parser.parse_args()
    np.random.seed(args.seed)

    env = DarwiNNEnvironment(cuda=False,seed=args.seed)
    args.popsize = (args.popsize // env.number_nodes) * env.number_nodes

    #define individual with its fitness interpretation function
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    #define function to be optimized
    toolbox.register("evaluate", benchmarks.rastrigin)

    #define update function
    strategy = cma.Strategy(centroid=[5.0]*args.dimension, sigma=5.0, lambda_=args.popsize)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    obj_fn = toolbox.evaluate
    update_fn = toolbox.update
    generate_fn = toolbox.generate

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)


    ne_optimizer = DEAPeaGenerateUpdateOptimizer(env, args.popsize, args.dimension, obj_fn, update_fn, generate_fn, hof, stats)

    for generation in range(1, args.generations + 1):
        ne_optimizer.step()
