import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from Datetime import date
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import seaborn as sns

def calculate_comprehensive_metrics(pred, gt, verbose=True):
    """
    Calculate PCC, Normalized RMSE, and Mutual Information
    
    Args:
        pred: Predicted values (numpy array or tensor)
        gt: Ground truth values (numpy array or tensor)
        verbose: Whether to print results
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Convert to numpy if needed
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    if hasattr(gt, 'cpu'):
        gt = gt.cpu().numpy()
    
    # Flatten arrays for calculation
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(pred_flat) | np.isnan(gt_flat))
    pred_clean = pred_flat[mask]
    gt_clean = gt_flat[mask]
    
    metrics = {}
    
    # 1. Pearson Correlation Coefficient
    if len(pred_clean) > 1:
        pcc, pcc_pval = pearsonr(pred_clean, gt_clean)
        metrics['PCC'] = pcc
        metrics['PCC_pval'] = pcc_pval
    else:
        metrics['PCC'] = np.nan
        metrics['PCC_pval'] = np.nan
    
    # 2. Normalized RMSE
    mse = np.mean((pred_clean - gt_clean) ** 2)
    rmse = np.sqrt(mse)
    gt_range = np.max(gt_clean) - np.min(gt_clean)
    normalized_rmse = rmse / gt_range if gt_range > 0 else np.nan
    metrics['NRMSE'] = normalized_rmse
    metrics['RMSE'] = rmse
    metrics['MSE'] = mse
    
    # 3. Mutual Information
    try:
        # For continuous variables, we can use mutual_info_regression
        # Reshape for sklearn
        pred_reshaped = pred_clean.reshape(-1, 1)
        mi = mutual_info_regression(pred_reshaped, gt_clean, random_state=42)[0]
        metrics['MI'] = mi
        
        # Normalized Mutual Information (using binning approach)
        # Bin the continuous variables for NMI calculation
        n_bins = min(50, len(pred_clean) // 10)  # Adaptive binning
        pred_binned = np.digitize(pred_clean, np.histogram(pred_clean, bins=n_bins)[1])
        gt_binned = np.digitize(gt_clean, np.histogram(gt_clean, bins=n_bins)[1])
        nmi = normalized_mutual_info_score(pred_binned, gt_binned)
        metrics['NMI'] = nmi
    except:
        metrics['MI'] = np.nan
        metrics['NMI'] = np.nan
    
    # Additional useful metrics
    metrics['MAE'] = np.mean(np.abs(pred_clean - gt_clean))
    metrics['R2'] = 1 - (np.sum((gt_clean - pred_clean) ** 2) / np.sum((gt_clean - np.mean(gt_clean)) ** 2))
    
    if verbose:
        print(f"Metrics Summary:")
        print(f"  PCC (Pearson):     {metrics['PCC']:.4f} (p-val: {metrics['PCC_pval']:.2e})")
        print(f"  NRMSE:             {metrics['NRMSE']:.4f}")
        print(f"  RMSE:              {metrics['RMSE']:.4f}")
        print(f"  MAE:               {metrics['MAE']:.4f}")
        print(f"  R²:                {metrics['R2']:.4f}")
        print(f"  Mutual Info:       {metrics['MI']:.4f}")
        print(f"  Normalized MI:     {metrics['NMI']:.4f}")
    
    
    return metrics

# TODO: Run deconvolution and save results for gt and pred. 
# TODO: Analyze gene spatial co-expression patterns in gt and pred.
# TODO: Analyze SVGs and gene expression patterns in gt and pred.

def save_results(model_name, date, epoch, genelist, metrics):
    filepath = '../model_results.csv'
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            f.write("id,Model,Date,Epoch,Genelist,PCC,NMI,NRMSE,RMSE,MAE,R2,MI\n")
            
    with open(filepath, 'a') as f:
        f.write(f"{model_name},{date},{epoch},{genelist},{metrics['PCC']:.4f},"
                f"{metrics['NMI']:.4f},{metrics['NRMSE']:.4f},{metrics['RMSE']:.4f},"
                f"{metrics['MAE']:.4f},{metrics['R2']:.4f},{metrics['MI']:.4f}\n")
        
def spatial_coexpression(pred):
    """
    Analyze spatial gene co-expression patterns using Giotto-like functionality
    """
    import matplotlib.pyplot as plt

    # Convert predictions to DataFrame format
    if hasattr(pred, 'cpu'):
        pred_np = pred.cpu().numpy()
    else:
        pred_np = np.array(pred)

    # Assume pred shape is (n_spots, n_genes)
    n_spots, n_genes = pred_np.shape

    # Create spatial coordinates (assuming grid layout)
    grid_size = int(np.sqrt(n_spots))
    x_coords = np.tile(np.arange(grid_size), grid_size)[:n_spots]
    y_coords = np.repeat(np.arange(grid_size), grid_size)[:n_spots]

    # Calculate pairwise gene correlations across spatial locations
    gene_correlations = np.corrcoef(pred_np.T)

    # Calculate spatial autocorrelation for each gene
    spatial_autocorr = []
    for gene_idx in range(n_genes):
        gene_expr = pred_np[:, gene_idx]
        
        # Create distance matrix between spots
        coords = np.column_stack([x_coords, y_coords])
        spatial_distances = squareform(pdist(coords))
        
        # Calculate Moran's I-like spatial autocorrelation
        weights = 1 / (1 + spatial_distances)  # Inverse distance weighting
        np.fill_diagonal(weights, 0)
        
        if np.sum(weights) > 0:
            mean_expr = np.mean(gene_expr)
            numerator = np.sum(weights * np.outer(gene_expr - mean_expr, gene_expr - mean_expr))
            denominator = np.sum(weights) * np.var(gene_expr)
            moran_i = numerator / denominator if denominator > 0 else 0
        else:
            moran_i = 0
        
        spatial_autocorr.append(moran_i)

    # Identify spatially co-expressed gene pairs
    coexpr_threshold = 0.7
    spatial_threshold = 0.3

    coexpressed_pairs = []
    for i in range(n_genes):
        for j in range(i+1, n_genes):
            if (abs(gene_correlations[i, j]) > coexpr_threshold and 
                spatial_autocorr[i] > spatial_threshold and 
                spatial_autocorr[j] > spatial_threshold):
                coexpressed_pairs.append((i, j, gene_correlations[i, j]))

    print(f"Found {len(coexpressed_pairs)} spatially co-expressed gene pairs")
    print(f"Average spatial autocorrelation: {np.mean(spatial_autocorr):.3f}")

    return {
        'gene_correlations': gene_correlations,
        'spatial_autocorrelation': spatial_autocorr,
        'coexpressed_pairs': coexpressed_pairs
    }