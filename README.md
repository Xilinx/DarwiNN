# DarwiNN

## Introduction

DarwiNN is a toolbox of functions enabling the training of DNN models using Evolutionary Strategies (ES), which we call Neuroevolution.
Unlike other evolutionary computation frameworks, DarwiNN is built on top of PyTorch, enabling easy integration into DNN training flows.
DarwiNN provides several GPU-accelerated evolutionary primitives - mutation, recombination - that enable fast execution of multiple flavors of ES, 
as well as support for distributed execution of neuroevolution.

## Features

### DNN Optimizers

DarwiNN provides a base abstract evolutionary DNN optimizer class from which several optimizers are derived:
- *OpenAIESOptimizer* which implements the DNN training algorithm developed by OpenAI for Reinforcement Learning and described in [this paper](https://arxiv.org/abs/1703.03864)
- *SNESOptimizer* which implements the SNES variant implemented for image classification in [this paper](https://arxiv.org/abs/1906.03139)
- *GAOptimizer*, a simple genetic algorithm DNN optimizer designed to resemble the one developed by Uber Research for Reinforcement learning and described in [this paper](https://arxiv.org/abs/1712.06567)

### Distribution Algorithm

DarwiNN can distribute the computation of any of its optimizers on an arbitrary number of GPUs, using the following distribution patterns:
- data-parallel inference with sequential evaluation of populaton individuals, which we call Distributed Data Parallel (DPP). 
In this distribution mode, parallelization is across the input batch, and is limited by the size of said batch. 
E.g. when training with batch size 64, the maximum number of GPUs is 64
- population-parallel evaluation, which we call Distributed Population Parallel (DPP), whereby the population is distributed across the available GPUs, and the maximum number of GPUs is equal to the population size.
- Semi-updates DPP, described [here](https://arxiv.org/abs/1906.03139), which more effectively distributes the gradient estimation step of ES
- Chromosome-updates DPP, which is similar to Semi-updates but minimizez communication requirements and scales better under network bandwidth constraints

### GPU-accelerated Coherent Distributed Mutation

Random noise is required in ES algorithms both for mutation and recombination. 
DarwiNN provides the NoiseGenerator class in `darwinn.utils.noise` which performs GPU-accelerated noise generation with specific slicing modes appropriate for each one of the distribution algorithms outlined above.

### GPU-accelerated ranking

Several ranking functions are provided in `darwinn.utils.fitness` - ranking, normalized ranking, centered ranking. 
These functions are implemented using Torch operators and are therefore fully GPU accelerated.
As these functions aren't typically very expensive computationally, the advantage of offloading them to GPU is that we don't have to move data back to the host to perform ranking.

## Getting Started

### Dependencies

DarwiNN depends on:
* Python (3.6+)
* MPI
* Pytorch (1.1.0+)
* DEAP (for specific black box optimization uses)

### Installation

```
git clone https://github.com/Xilinx/DarwiNN.git
cd darwinn
pip install -e .
```

To learn how to use DarwiNN for DNN training, please refer to the image classification examples in the `examples` folder. 
