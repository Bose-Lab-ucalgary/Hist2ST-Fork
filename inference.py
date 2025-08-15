"""
Simple test to validate a single Hist2ST model
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import scanpy as sc
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
import scipy.stats as stats

# Set environment variable for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from dataset import ViT_HEST1K
from config import GENE_LISTS
from HIST2ST import Hist2ST
from predict import test

def create_hist2st_model(tag="5-7-2-8-4-16-32", gene_list="HER2ST"):
    """Create and configure the Hist2ST model from tag string."""
    # Unpack model hyperparameters from tag string
    k, p, d1, d2, d3, h, c = map(lambda x: int(x), tag.split('-'))
    
    # Get number of genes from config
    n_genes = GENE_LISTS[gene_list]["n_genes"]
    
    model = Hist2ST(
        depth1=d1, depth2=d2, depth3=d3, n_genes=n_genes,
        kernel_size=k, patch_size=p,
        heads=h, channel=c, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5,
    )
    
    return model, n_genes

def test_model(checkpoint_path="./model/5-Hist2ST.ckpt", 
               tag="5-7-2-8-4-16-32", 
               gene_list="HER2ST",
               device="cuda"):
    """Test validation of a single Hist2ST model and save aligned predictions + ground truth"""
    
    print("Testing Hist2ST model validation...")
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
  
    try:
        # Create model
        print(f"Creating Hist2ST model with tag: {tag}")
        model, n_genes = create_hist2st_model(tag=tag, gene_list=gene_list)
        print(f"✓ Model created with {n_genes} genes")
        
        # Load model weights
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        print("✓ Model loaded and moved to device")
        
        # Create dataset
        print("Creating dataset...")
        dataset = ViT_HEST1K(mode='All', gene_list=gene_list, cancer_only=True)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Check if dataset is empty
        if len(dataset) == 0:
            print("ERROR: Dataset is empty!")
            return
        
        # Run predictions using the Hist2ST test function
        print("Running predictions...")
        pred, gt = test(model, data_loader, device)
        print(f"✓ Predictions completed")
        
        # Extract data from AnnData objects
        if hasattr(pred, 'X') and hasattr(gt, 'X'):
            all_preds = pred.X if not hasattr(pred.X, 'toarray') else pred.X.toarray()
            all_gt = gt.X if not hasattr(gt.X, 'toarray') else gt.X.toarray()
        else:
            all_preds = np.array(pred) if not isinstance(pred, np.ndarray) else pred
            all_gt = np.array(gt) if not isinstance(gt, np.ndarray) else gt
        
        # GET SAMPLE IDs FROM ADATA OBJECT
        if hasattr(pred, 'obs') and 'sample_id' in pred.obs:
            sample_ids = pred.obs['sample_id'].tolist()
            print(f"✓ Got {len(sample_ids)} sample IDs from AnnData object")
        else:
            # Fallback to dataset sample IDs
            sample_ids = dataset.sample_ids.copy()
            print(f"⚠ Fallback: Got {len(sample_ids)} sample IDs from dataset")
        
        print(f"✓ Processed data - Predictions: {all_preds.shape}, Ground truth: {all_gt.shape}")
        
        # Validate alignment
        if len(sample_ids) != all_preds.shape[0]:
            print(f"WARNING: Sample ID count ({len(sample_ids)}) doesn't match prediction count ({all_preds.shape[0]})")
            # Trim to match
            min_len = min(len(sample_ids), all_preds.shape[0])
            sample_ids = sample_ids[:min_len]
            all_preds = all_preds[:min_len]
            all_gt = all_gt[:min_len]
            print(f"✓ Trimmed to {min_len} samples for alignment")
        
        # Save raw data only
        output_data = {
            'predictions': all_preds,
            'ground_truth': all_gt,
            'sample_ids': sample_ids
        }

        # Create output directory if it doesn't exist
        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)

        # Save as numpy file with raw data only
        output_path = os.path.join(output_dir, f"{gene_list}_pred_gt.npz")
        np.savez(output_path, **output_data)
        print(f"✓ Raw data saved to {output_path}")
        
        # Print basic summary
        print(f"\nSummary:")
        print(f"- Model: Hist2ST")
        print(f"- Gene list: {gene_list} ({n_genes} genes)")
        print(f"- Samples processed: {len(sample_ids)}")
        print(f"- Predictions shape: {all_preds.shape}")
        print(f"- Ground truth shape: {all_gt.shape}")
        print(f"- Sample IDs: {sample_ids[:5]}..." if len(sample_ids) > 5 else f"- Sample IDs: {sample_ids}")
        
        return all_preds, all_gt, sample_ids
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
        
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def test_model_chunked(checkpoint_path="./model/5-Hist2ST.ckpt", 
                      tag="5-7-2-8-4-16-32", 
                      gene_list="HER2ST",
                      device="cuda",
                      max_spots_per_chunk=5000):  # Add chunk size limit
    
    print("Testing Hist2ST model validation (chunked)...")
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
  
    try:
        # Create model
        print(f"Creating Hist2ST model with tag: {tag}")
        model, n_genes = create_hist2st_model(tag=tag, gene_list=gene_list)
        print(f"✓ Model created with {n_genes} genes")
        
        # Load model weights
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model weights from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        print("✓ Model loaded and moved to device")
        
        # Create dataset
        print("Creating dataset...")
        dataset = ViT_HEST1K(mode='All', gene_list=gene_list, cancer_only=True)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Check if dataset is empty
        if len(dataset) == 0:
            print("ERROR: Dataset is empty!")
            return
        
        # Process samples one by one
        all_predictions = []
        all_ground_truth = []
        all_sample_ids = []
        
        print(f"Processing {len(dataset)} samples individually...")
        
        for i in tqdm(range(len(dataset))):
            try:
                # Clear GPU memory before each sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create single-sample loader
                single_sample_dataset = torch.utils.data.Subset(dataset, [i])
                single_loader = DataLoader(single_sample_dataset, batch_size=1, num_workers=0)
                
                # Get sample info
                sample_data = dataset[i]
                n_spots = sample_data[0].shape[0] if hasattr(sample_data[0], 'shape') else len(sample_data[0])
                sample_id = dataset.sample_ids[i] if hasattr(dataset, 'sample_ids') else f"sample_{i}"
                
                print(f"Processing sample {sample_id} with {n_spots} spots...")
                
                # Skip if too large
                if n_spots > max_spots_per_chunk:
                    print(f"⚠ Skipping sample {sample_id} - too large ({n_spots} > {max_spots_per_chunk})")
                    continue
                
                # Run prediction for this sample
                with torch.no_grad():
                    sample_pred, sample_gt = test(model, single_loader, device)
                    
                    # Extract data
                    if hasattr(sample_pred, 'X'):
                        pred_data = sample_pred.X if not hasattr(sample_pred.X, 'toarray') else sample_pred.X.toarray()
                        gt_data = sample_gt.X if not hasattr(sample_gt.X, 'toarray') else sample_gt.X.toarray()
                    else:
                        pred_data = np.array(sample_pred)
                        gt_data = np.array(sample_gt)
                    
                    all_predictions.append(pred_data)
                    all_ground_truth.append(gt_data)
                    
                    # Add sample IDs for each spot
                    if hasattr(sample_pred, 'obs') and 'sample_id' in sample_pred.obs:
                        sample_ids_list = sample_pred.obs['sample_id'].tolist()
                    else:
                        sample_ids_list = [sample_id] * pred_data.shape[0]
                    
                    all_sample_ids.extend(sample_ids_list)
                    
                    print(f"✓ Completed sample {sample_id}: {pred_data.shape}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠ Skipping sample {sample_id} - out of memory")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Concatenate all results
        if all_predictions:
            final_predictions = np.vstack(all_predictions)
            final_ground_truth = np.vstack(all_ground_truth)
            
            print(f"✓ Processed {len(all_predictions)} samples successfully")
            print(f"Final shapes: predictions {final_predictions.shape}, ground truth {final_ground_truth.shape}")
            
            # Save results
            output_data = {
                'predictions': final_predictions,
                'ground_truth': final_ground_truth,
                'sample_ids': all_sample_ids
            }
            
            output_dir = "./results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{gene_list}_pred_gt.npz")
            np.savez(output_path, **output_data)
            print(f"✓ Results saved to {output_path}")
            
            return final_predictions, final_ground_truth, all_sample_ids
        else:
            print("❌ No samples were processed successfully")
            return None, None, None
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
        
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Test with different model configurations
    print("Starting Hist2ST inference test...")
    
    # Configuration options
    checkpoint_path = "./model/5-Hist2ST.ckpt"
    tag = "5-7-2-8-4-16-32"  # Model architecture tag
    gene_list = "HER2ST"  # Gene list to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Configuration:")
    print(f"- Checkpoint: {checkpoint_path}")
    print(f"- Model tag: {tag}")
    print(f"- Gene list: {gene_list}")
    print(f"- Device: {device}")
    print("-" * 50)
    
    # test_model(
    #     checkpoint_path=checkpoint_path,
    #     tag=tag,
    #     gene_list=gene_list,
    #     device=device
    # )
    
    test_model_chunked(
        checkpoint_path=checkpoint_path,
        tag=tag,
        gene_list=gene_list,
        device=device,
        max_spots_per_chunk=8000  # Adjust based on your GPU memory
    )
    print("Hist2ST inference test completed successfully.")

