#!/usr/bin/env python3
"""
Simple Hist2ST Inference Script
A simplified version for running inference with the trained model.
"""

import torch
import numpy as np
from predict import *
from HIST2ST import *
from dataset import ViT_HER2ST, ViT_SKIN
from torch.utils.data import DataLoader
import argparse
import os


def simple_inference(fold=5, dataset='her2st', device='cuda'):
    """Run basic inference with minimal configuration."""
    
    # Model configuration (from your notebook)
    tag = '5-7-2-8-4-16-32'
    k, p, d1, d2, d3, h, c = map(lambda x: int(x), tag.split('-'))
    
    # Dataset specific parameters
    genes = 171 if dataset == 'cscc' else 785
    prune = 'Grid' if dataset == 'her2st' else 'NA'
    
    print(f"Running inference for fold {fold} on {dataset} dataset")
    print(f"Device: {device}")
    print(f"Model config: k={k}, p={p}, d1={d1}, d2={d2}, d3={d3}, h={h}, c={c}")
    
    # Load test dataset
    print("Loading test dataset...")
    testset = pk_load(fold, 'test', dataset=dataset, flatten=False, adj=True, ori=True, prune=prune)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    print(f"Test dataset loaded: {len(testset)} samples")
    
    # Create model
    print("Creating model...")
    model = Hist2ST(
        depth1=d1, depth2=d2, depth3=d3, n_genes=genes,
        kernel_size=k, patch_size=p,
        heads=h, channel=c, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5,
    )
    
    # Load model checkpoint
    checkpoint_path = f'./model/{fold}-Hist2ST.ckpt'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return None, None
    
    print(f"Loading model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        pred, gt = test(model, test_loader, device)
    
    # Calculate Pearson correlation
    R = get_R(pred, gt)[0]
    pearson_corr = np.nanmean(R)
    print(f'Pearson Correlation: {pearson_corr:.4f}')
    
    return pred, gt


def main():
    parser = argparse.ArgumentParser(description='Simple Hist2ST inference')
    parser.add_argument('--fold', type=int, default=5, help='Fold number (default: 5)')
    parser.add_argument('--dataset', type=str, default='her2st', choices=['her2st', 'cscc'], 
                       help='Dataset type (default: her2st)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run inference
    try:
        pred, gt = simple_inference(args.fold, args.dataset, args.device)
        if pred is not None:
            print("Inference completed successfully!")
            print(f"Prediction shape: {pred.shape}")
            print(f"Ground truth shape: {gt.shape}")
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
