#!/bin/bash -l 

#SBATCH -t 1:00:00
#SBATCH --nodelist=xirxlabs51

WORKDIR=/scratch/users/lucian/DarwiNN
SIMG=$WORKDIR/darwinn.sif

echo "Running DarwiNN"
mpirun -np 3 --oversubscribe -x MASTER_ADDR=xirxlabs51 -x MASTER_PORT=23456 singularity exec --nv $SIMG python $WORKDIR/examples/mnist.py --epochs 1 --verbose --lr 0.001 --popsize 96
