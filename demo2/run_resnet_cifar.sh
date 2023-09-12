#!/bin/zsh 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=test
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=selma.gudmundsen@gmail.com
#SBATCH -o test_out.txt
#SBATCH -e test_err.txt
#SBATCH --partition=test
#SBATCH --gres=gpu:1

conda activate env1

python cifar10.py