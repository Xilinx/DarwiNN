# DarwiNN
Distributed neuroevolution framework based on Pytorch.

# Requirements

## Dependencies

DarwiNN depends on:
* MPI
* Pytorch distributed MPI backend

## Docker

A Dockerfile is available under `images/Dockerfile`. To build an image run:

`docker build -t darwinn:latest -f images/Dockerfile .`

# Examples

Example scripts are provided in `examples/`, which train various NNs for MNIST (using a small 2-layer CNN) and CIFAR10 (using LeNet, Caffe's CIFAR10 Quick, or Network-in-Network). 

To quickly run these examples:

```
docker run -it --rm -v `pwd`:/work --workdir /work darwinn:latest mpirun --allow-run-as-root -np 1 python examples/[mnist/cifar10].py --epochs 1 --lr 0.001 --no-cuda --population 5
```

For a detailed list of options available run:

```
docker run -it --rm -v `pwd`:/work --workdir /work darwinn:latest mpirun --allow-run-as-root -np 1 python examples/[mnist/cifar10].py --help
```

# Profiling

The Docker image built from the provided Dockerfile contains an installation of the [TAU](https://www.cs.uoregon.edu/research/tau/home.php) profiler.
To profile any DarwiNN run with TAU, simply execute with `tau_python` instead of `python`:

```
docker run -it --rm -v `pwd`:/work --workdir /work darwinn:latest mpirun --allow-run-as-root -np 1 tau_python examples/mnist.py --epochs 1 --lr 0.001 --no-cuda --population 5
docker run -it --rm -v `pwd`:/work --workdir /work darwinn:latest pprof
```
