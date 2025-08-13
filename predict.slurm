#!/bin/bash

####### Reserve computing resources #############
#SBATCH --job-name=hist2st-predict
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100

####### Run your script #########################
source ~/software/init-conda
conda activate hist2st-env

python inference.py
# python debug_adjacency.py