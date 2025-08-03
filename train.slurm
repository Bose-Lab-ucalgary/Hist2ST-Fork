#!/bin/bash

####### Reserve computing resources #############
#SBATCH --job-name=hist2st
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100

####### Run your script #########################
source ~/software/init-conda
conda activate hist2st-env

# Example configurations - uncomment the one you want to use

# Single GPU training with 3CA gene list
# python HEST_train.py --mode train_test --dataset HEST1K --gene_list 3CA --gpus 1 --num_workers 16 --epochs 300

# Single GPU training with 3CA gene list, small epoch
python HEST_train.py --mode train_test --dataset HEST1K --gene_list 3CA --gpus 1 --num_workers 8 --epochs 5

# Multi-GPU training with DDP strategy, small epoch
# python HEST_train.py --mode train_test --dataset HEST1K --gene_list 3CA --gpus 2 --strategy ddp --num_workers 16 --epochs 5

# Multi-GPU training with DDP strategy
# python HEST_train.py --mode train_test --dataset HEST1K --gene_list 3CA --gpus 2 --strategy ddp --num_workers 16 --epochs 300

# Training only (no testing)
# python HEST_train.py --mode train --dataset HEST1K --gene_list 3CA --gpus 2 --num_workers 16 --epochs 300

# Testing only with specific checkpoint
# python HEST_train.py --mode test --dataset HEST1K --gene_list 3CA --gpus 1 --num_workers 8 --checkpoint_path model/specific_model.ckpt

# All phases (train, validate, test)
# python HEST_train.py --mode all --dataset HEST1K --gene_list 3CA --gpus 2 --strategy ddp --num_workers 16 --epochs 300
