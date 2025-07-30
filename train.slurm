#!/bin/bash

####### Reserve computing resources #############
#SBATCH --job-name=hist2st
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=140G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu-a100

####### Run your script #########################
source ~/software/init-conda
conda activate hist2st-env


python train.py