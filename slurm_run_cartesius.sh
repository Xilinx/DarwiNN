#!/bin/bash -l 

#SBATCH -t 1:00:00
#SBATCH --partition=gpu_short
#SBATCH --nodes=4

SIMG=/home/xisurflp/darwinn.sif

module purge
module load mpi/openmpi/3.1.2-cuda10

#run in population-parallel
time mpirun --map-by ppr:2:node -np 8 -x NCCL_SOCKET_IFNAME=enp5s0f0 -x MASTER_ADDR=$SLURMD_NODENAME -x MASTER_PORT=24567 singularity exec --nv /home/xisurflp/darwinn.sif python examples/mnist.py --verbose --epochs 1 --popsize 960 --lr 0.004 --batch-size 1024
#run in data-parallel
time mpirun --map-by ppr:2:node -np 8 -x NCCL_SOCKET_IFNAME=enp5s0f0 -x MASTER_ADDR=$SLURMD_NODENAME -x MASTER_PORT=24567 singularity exec --nv /home/xisurflp/darwinn.sif python examples/mnist.py --verbose --epochs 1 --popsize 960 --lr 0.004 --batch-size 128 --ddp
