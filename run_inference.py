"""
Hist2ST-Fork Inference Script
This script runs inference/testing for the trained Hist2ST model.
"""

import torch
import numpy as np
import pandas as pd
import os
import random
import argparse
from tqdm import tqdm
import yaml
from predict import *
from HIST2ST import *
from dataset import ViT_HER2ST, ViT_SKIN
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def set_seed(seed=12000):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Hist2ST inference')
    parser.add_argument('--fold', type=int, default=5, 
                       help='Fold number for model checkpoint (default: 5)')
    parser.add_argument('--dataset', type=str, default='her2st', 
                       choices=['her2st', 'cscc'], 
                       help='Dataset type (default: her2st)')
    parser.add_argument('--model_dir', type=str, default='./model', 
                       help='Directory containing model checkpoints (default: ./model)')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference (default: cuda)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for inference (default: 1)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of workers for data loading (default: 0)')
    parser.add_argument('--tag', type=str, default='5-7-2-8-4-16-32',
                       help='Model configuration tag (default: 5-7-2-8-4-16-32)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save inference results (default: ./results)')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction results to file')
    parser.add_argument('--calculate_ari', action='store_true',
                       help='Calculate ARI score (requires clustering)')
    
    return parser.parse_args()


def load_conversion_dict(conversion_path):
    """Load gene ID conversion dictionaries."""
    try:
        conversion = pd.read_csv(conversion_path, sep="\t").loc[:, ['ensembl_gene_id', 'symbol']]
        convert_to_ens = conversion.set_index('symbol')['ensembl_gene_id'].to_dict()
        convert_to_sym = conversion.set_index('ensembl_gene_id')['symbol'].to_dict()
        return convert_to_ens, convert_to_sym
    except FileNotFoundError:
        print(f"Warning: Conversion file not found at {conversion_path}")
        return {}, {}


def emb_to_sym(label, convert_to_sym, convert_to_ens):
    """Convert the label to a list of gene symbols."""
    if isinstance(label, str):
        for gene in label.split(' '):
            if gene not in convert_to_sym and gene in convert_to_ens:
                return label
            elif gene not in convert_to_sym:
                print(f"Error: {gene} not in conversion dictionary.")
            else:
                conversion = [convert_to_sym[gene] for gene in label.split(' ')]
                return ' '.join(conversion)
    elif isinstance(label, list):
        conversion = [convert_to_sym[gene] for gene in label]
        return ' '.join(conversion)
    else:
        return []


def create_model(tag, genes, device):
    """Create and configure the Hist2ST model."""
    # Unpack model hyperparameters from tag string
    k, p, d1, d2, d3, h, c = map(lambda x: int(x), tag.split('-'))
    
    model = Hist2ST(
        depth1=d1, depth2=d2, depth3=d3, n_genes=genes,
        kernel_size=k, patch_size=p,
        heads=h, channel=c, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5,
    )
    
    return model, (k, p, d1, d2, d3, h, c)

def create_model(path):
    """Create and configure the Hist2ST model."""
    # Unpack model hyperparameters from a yaml file
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    bake = config['bake']
    channel = config['channel']
    depth1 = config['depth1']
    depth2 = config['depth2']
    depth3 = config['depth3']
    dropout = config['dropout']
    fig_size = config['fig_size']
    heads = config['heads']
    kernel_size = config['kernel_size']
    lamb = config['lamb']
    n_genes = config['n_genes']
    n_pos = config['n_pos']
    nb = config['nb']
    patch_size = config['patch_size']
    policy = config['policy']
    zinb = config['zinb']
    
    model = Hist2ST(
        depth1=depth1, depth2=depth2, depth3=depth3,
        n_genes=n_genes, learning_rate=config['learning_rate'],
        kernel_size=kernel_size, patch_size=patch_size,
        heads=heads, channel=channel, dropout=dropout,
        zinb=zinb, nb=nb,
        bake=bake, lamb=lamb,
        n_pos=n_pos, policy=policy
    )
    
    return model, (kernel_size, patch_size, depth1, depth2, depth3, heads, channel)

def load_model_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_inference(model, test_loader, device):
    """Run inference on the test dataset."""
    print("Running inference...")
    with torch.no_grad():
        pred, gt = test(model, test_loader, device)
    return pred, gt


def calculate_metrics(pred, gt, label=None, calculate_ari=False):
    """Calculate evaluation metrics."""
    results = {}
    
    # Calculate Pearson correlation
    R = get_R(pred, gt)[0]
    pearson_corr = np.nanmean(R)
    results['pearson_correlation'] = pearson_corr
    print(f'Pearson Correlation: {pearson_corr:.4f}')
    
    # Calculate ARI if requested and label is available
    if calculate_ari and label is not None:
        try:
            clus, ARI = cluster(pred, label)
            results['ari_score'] = ARI
            print(f'ARI Score: {ARI:.4f}')
        except Exception as e:
            print(f"Warning: Could not calculate ARI score: {e}")
            results['ari_score'] = None
    
    return results


def save_results(pred, gt, results, output_dir, fold, dataset):
    """Save inference results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions and ground truth
    pred_file = os.path.join(output_dir, f'fold_{fold}_{dataset}_predictions.npy')
    gt_file = os.path.join(output_dir, f'fold_{fold}_{dataset}_ground_truth.npy')
    
    np.save(pred_file, pred)
    np.save(gt_file, gt)
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f'fold_{fold}_{dataset}_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Inference Results for Fold {fold} - Dataset {dataset}\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to {output_dir}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    print(f"Running inference for fold {args.fold} on {args.dataset} dataset")
    
    # Configure dataset-specific parameters
    prune = 'Grid' if args.dataset == 'her2st' else 'NA'
    genes = 171 if args.dataset == 'cscc' else 785
    
    # Load gene conversion dictionaries (optional)
    conversion_path = '../../../../tahsin/HEST/assets/gene_ids/hgnc_complete_set.txt'
    convert_to_ens, convert_to_sym = load_conversion_dict(conversion_path)
    
    try:
        # Load test dataset
        print("Loading test dataset...")
        testset = pk_load(args.fold, 'test', dataset=args.dataset, 
                         flatten=False, adj=True, ori=True, prune=prune)
        test_loader = DataLoader(testset, batch_size=args.batch_size, 
                               num_workers=args.num_workers, shuffle=False)
        
        # Get label for ARI calculation
        label = testset.label[testset.names[0]] if hasattr(testset, 'label') and len(testset.names) > 0 else None
        
        print(f"Test dataset loaded: {len(testset)} samples")
        
        # Create model
        print("Creating model...")
        model, hyperparams = create_model(args.tag, genes, device)
        k, p, d1, d2, d3, h, c = hyperparams
        print(f"Model configuration - k:{k}, p:{p}, d1:{d1}, d2:{d2}, d3:{d3}, h:{h}, c:{c}")
        
        # Load model checkpoint
        checkpoint_path = os.path.join(args.model_dir, f'{args.fold}-Hist2ST.ckpt')
        model = load_model_checkpoint(model, checkpoint_path, device)
        
        # Run inference
        pred, gt = run_inference(model, test_loader, device)
        
        # Calculate metrics
        results = calculate_metrics(pred, gt, label, args.calculate_ari)
        
        # Save results if requested
        if args.save_predictions:
            save_results(pred, gt, results, args.output_dir, args.fold, args.dataset)
        
        print("\nInference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
