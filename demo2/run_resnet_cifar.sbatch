#!/bin/bin
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=cifar10

# task per node
##SBATCH --ntasks 1

# cores per node
##SBATCH --cpus-per-task 8

# memory per core
##SBATCH --mem=16gb

# total runtime, enable to use GPUs with max times
#SBATCH --time=3:00:00 #3 hours

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=All
#SBATCH --mail-user=selma.gudmundsen@gmail.com

# output file
#SBATCH --output=cifar_resnet_1.out

#conda init
source/home/Student/${USER}/.bashrc

# load modules
#JAX CUDA 11.1 Setup
#~module load cuda/11.1
#activate venv
#~source~/jax-venv-/bin/activate
conda activate COMP3710

#go to data directory
cd /${USER}/dev/COMP3710/COMP3710/demo2

#print python version
python --version

#run script
python cifar10.py
