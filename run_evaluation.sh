#!/bin/bash

#SBATCH --job-name=ACDCevaluation
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8

source /home/${USER}/.bashrc
# source /home/${USER}/miniconda3/bin/activate
source /home/${USER}/anaconda/bin/activate
source activate simclr # conda -> source

srun python linear_evaluation.py

