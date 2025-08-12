import torch
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl 
from torch.utils.data import DataLoader
from measure import calculate_comprehensive_metrics, save_results
from predict import test
from config import GENE_LISTS
from HIST2ST import Hist2ST
from predict import get_R
from run_inference import load_model_checkpoint
from dataset import ViT_HEST1K


def run_test_phase(model, vit_dataset=ViT_HEST1K, model_address="model", dataset_name="hest1k", gene_list="3CA", checkpoint_path=None, num_workers=16, gpus=1, neighbors=5, prune="NA", batch_size=1):
    n_genes = GENE_LISTS[gene_list]["n_genes"]
    # Load test dataset
    testset = vit_dataset(mode='test', flatten=False, adj=True, ori=True, prune=prune, neighs=neighbors, gene_list=gene_list)
    test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)

    # Load model
    if checkpoint_path is None:
        checkpoint_path = os.path.join(model_address, f"last_train_{gene_list}_{dataset_name}_{n_genes}.ckpt")
    print(f"Loading model from: {checkpoint_path}")
    
    model = load_model_checkpoint(model, checkpoint_path)
    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        logger=False,
        enable_progress_bar=False,
        )
    
    predictions = trainer.predict(model, test_loader)
        
    # Process predictions
    all_preds = []
    all_gts = []
    all_sample_info = []
    
    for i, batch_result in enumerate(predictions):
        pred = batch_result['pred']
        gt = batch_result['gt']
        batch_idx = batch_result.get('batch_idx', i)
        sample_id = batch_result.get('sample_id', f"sample_{i}")
        centers = batch_result['centers']
        
        all_preds.append(pred)
        all_gts.append(gt)
        all_sample_info.append({
            'batch_idx': batch_idx,
            'sample_id': sample_id,
            'pred_shape': pred.shape,
            'gt_shape': gt.shape,
            'centers': centers
        })
    # Concatenate all predictions
    pred_array = np.concatenate(all_preds, axis=0)
    gt_array = np.concatenate(all_gts, axis=0)
    
    print(f"\nFinal arrays - Predictions: {pred_array.shape}, Ground truth: {gt_array.shape}")
    
    # Calculate comprehensive metrics
    print("\n" + "="*30)
    print("TEST METRICS")
    print("="*30)
    metrics = calculate_comprehensive_metrics(pred_array, gt_array, verbose=True)
    meta_df = get_meta()
    
    # Legacy compatibility
    R, p_val = get_R(pred_array, gt_array)[0]
    print(f'Legacy Test Pearson Correlation: {np.nanmean(R):.4f}')
    
    # Return predictions with sample info
    return {
        'predictions': pred_array,
        'ground_truth': gt_array,
        'metrics': metrics,
        'sample_info': all_sample_info,
        'R': R,
        'p_val': p_val
    }
    
def get_meta(results_path = '../model_results.csv'):
    meta_hest_path = "/work/bose_lab/tahsin/data/HEST"

    meta_df = pd.read_csv(os.path.join(meta_hest_path, "HEST_v1_1_0.csv"))
    meta_df = meta_df[meta_df['species'] == 'Homo sapiens']
    
    return meta_df
    # results_df = pd.read_csv(results_path)
    
    # total_df = pd.merge(results_df, meta_df, )
    
    # meta_df = meta_df[['id', 'organ', 'disease_state', 'oncotree_code', 'patient', 'st_technology', 'dat']]

    # id = meta_df['id']
    # image_filename = meta_df['image_filename']
    # organ = meta_df['organ']
    # disease_state = meta_df['disease_state']
    # oncotree_code = meta_df['oncotree_code']
    # species = meta_df['species']
    # patient = meta_df['patient']
    # st_technology = meta_df['st_technology']
    # data_publication_date = meta_df['data_publication_date']
    # license = meta_df['license']
    # study_link = meta_df['study_link']
    # download_page_link1 = meta_df['download_page_link1']
    # inter_spot_dist = meta_df['inter_spot_dist']
    # spot_diameter = meta_df['spot_diameter']
    # spots_under_tissue = meta_df['spots_under_tissue']
    # preservation_method = meta_df['preservation_method']
    # nb_genes = meta_df['nb_genes']
    # treatment_comment = meta_df['treatment_comment']
    # tissue = meta_df['tissue']
    # disease_comment = meta_df['disease_comment']
    # subseries = meta_df['subseries']