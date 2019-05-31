# DarwiNN
Distributed neuroevolution framework with really poor documentation.

# Requirements

## Dependencies

DarwiNN depends on:
* MPI
* Pytorch distributed MPI backend

## Docker

A Dockerfile is available under `images/Dockerfile`. To build an image run:

`docker build -t darwinn:latest -f images/Dockerfile .`

# Run a MNIST Example

An example is provided in `examples/conv2.py`, which trains a small CNN for MNIST. To run this example:

`docker run -it --rm -v `pwd`:/work --workdir /work darwinn:latest mpirun --allow-run-as-root -np 1 python examples/conv2.py --epochs 1 --lr 0.001 --no-cuda --population 5`

# Profiling

The Docker image built from the provided Dockerfile contains an installation of the [TAU](https://www.cs.uoregon.edu/research/tau/home.php) profiler.
To profile any DarwiNN run with TAU, simply execute with `tau_python` instead of `python`:

```
docker run -it --rm -v `pwd`:/work --workdir /work darwinn:latest mpirun --allow-run-as-root -np 1 tau_python examples/conv2.py --epochs 1 --lr 0.001 --no-cuda --population 5
docker run -it --rm -v `pwd`:/work --workdir /work darwinn:latest pprof
```
