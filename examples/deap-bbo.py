import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import benchmarks
from deap import cma
from deap import algorithms
import darwinn as dwn
import argparse
from optimizers.deap import DEAPeaGenerateUpdateOptimizer

if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='DarwiNN distributed DEAP Example (Rastrigin+CMA-ES)')
    parser.add_argument('--dimension', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--generations', type=int, default=100, metavar='N',
                        help='how many generations to run')
    args = parser.parse_args()
    args.popsize = 100*args.dimension
    np.random.seed(args.seed)

    env = dwn.DarwiNNEnvironment(cuda=False,seed=args.seed)
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
