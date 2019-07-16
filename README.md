# DarwiNN
Distributed neuroevolution framework based on Pytorch.

# Requirements

## Dependencies

DarwiNN depends on:
* Python (3.6+)
* MPI
* Pytorch (1.1.0+)
* DEAP (for specific black box optimization uses)

## Installation

### Pip
DarwiNN is pip-installable with the following commands:

```
python setup.py bdist_wheel
pip install dist/*.whl
```

### Docker

To build a Docker image for DarwiNN run:

`docker build -t darwinn:latest .`

# Examples

Example scripts are provided in `examples/`, which train various NNs for MNIST (using a small 2-layer CNN) and CIFAR10 (using LeNet, Caffe's CIFAR10 Quick, or Network-in-Network). 

To quickly run these examples:

```
docker run -it --rm --ipc host -v `pwd`:/work --workdir /work darwinn:latest mpirun --allow-run-as-root -np 1 python examples/[mnist/cifar10].py --epochs 1 --lr 0.001 --no-cuda --popsize 5
```

For a detailed list of options available run:

```
docker run -it --rm -v `pwd`:/work --workdir /work darwinn:latest python examples/[mnist/cifar10].py --help
```
