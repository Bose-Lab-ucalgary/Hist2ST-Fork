import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from dataset import ViT_HER2ST, ViT_SKIN, ViT_HEST1K
from scipy.stats import pearsonr,spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
def pk_load(fold,mode='train', flatten=False, dataset='her2st',r=4,ori=True,adj=True,prune='Grid',neighs=4, genelist = None):
    assert dataset in ['her2st', 'cscc', 'hest1k']
    
    # Debug dataset parameters
    print(f"\nLoading dataset with parameters:")
    print(f"  Dataset: {dataset}")
    print(f"  Mode: {mode}")
    print(f"  Fold: {fold}")
    
    if dataset == 'hest1k':
        dataset = ViT_HEST1K(
            mode=mode, 
            flatten=flatten,
            ori=ori,
            neighs=neighs,
            adj=adj,
            prune=prune,
            r=r,
            gene_list=genelist,
        )
        # Verify dataset loading
        print(f"\nDataset loaded:")
        print(f"  Length: {len(dataset)}")
        if hasattr(dataset, 'adata'):
            print(f"  AnnData shape: {dataset.adata.shape}")
            print(f"  Index unique: {dataset.adata.obs_names.is_unique}")
    
    elif dataset=='her2st':
        dataset = ViT_HER2ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    elif dataset=='cscc':
        dataset = ViT_SKIN(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    # elif dataset=='hest1k':
    #     dataset = ViT_HEST1K(
    #         mode=mode,fold=fold,flatten=flatten,sample_ids=sample_ids,
    #         ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
    #     )
    return dataset



def test(model, adata_loader, device):
    model.eval()
    
    pred_list = []
    gt_list = []
    coords_list = []
    sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(adata_loader):
            # Debug the batch unpacking
            print(f"Batch length: {len(batch)}")
            for i, component in enumerate(batch):
                if hasattr(component, 'shape'):
                    print(f"  Batch component {i}: {component.shape}")
            
            patch, position, exp, adj, *_, centers, sample_id = batch
            
            # Move to device
            patch = patch.to(device)                    
            positions = position.to(device)   
            exp = exp.squeeze(0).to(device)             
            adj = adj.squeeze(0).to(device)             
            centers = centers.squeeze(0).to(device)                
            
            # In your inference function, check which tensor you're passing:
            print(f"positions range: [{positions.min()}, {positions.max()}]")  # Should be [0, 63]
            print(f"centers range: [{centers.min()}, {centers.max()}]")        # Might be large pixel coords

            # Make sure you're passing positions, not centers to the model:
            pred = model(patch, positions, adj)[0]  # Use positions, not centers
            # Model call
            # pred = model(patch, centers, adj)[0]
            
            # Convert to numpy and append
            preds = pred.squeeze().cpu().numpy()
            ct = centers.squeeze().cpu().numpy()
            gt = exp.squeeze().cpu().numpy()
            
            # Get number of spots for this sample
            n_spots = preds.shape[0]
            
            # Create sample IDs for each spot
            if isinstance(sample_id, (list, tuple)):
                current_sample_id = sample_id[0]
            elif isinstance(sample_id, str):
                current_sample_id = sample_id
            else:
                current_sample_id = str(sample_id)
            
            # Add data to lists
            pred_list.append(preds)
            gt_list.append(gt)
            coords_list.append(ct)
            sample_ids.extend([current_sample_id] * n_spots)  # Extend, not append!
    
    # FIX: CONCATENATE instead of trying to create array from variable-sized arrays
    all_preds = np.concatenate(pred_list, axis=0)      
    all_gt = np.concatenate(gt_list, axis=0)           
    all_coords = np.concatenate(coords_list, axis=0)   
    
    # Create AnnData objects with concatenated data
    adata = ad.AnnData(all_preds)
    adata.obsm['spatial'] = all_coords
    adata.obs['sample_id'] = sample_ids
    
    adata_gt = ad.AnnData(all_gt)
    adata_gt.obsm['spatial'] = all_coords
    adata_gt.obs['sample_id'] = sample_ids
    
    return adata, adata_gt

def cluster(adata,label):
    idx=label!='undetermined'
    tmp=adata[idx]
    l=label[idx]
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p=kmeans.labels_.astype(str)
    lbl=np.full(len(adata),str(len(set(l))))
    lbl[idx]=p
    adata.obs['kmeans']=lbl
    return p,round(ari_score(p,l),3)
def get_R(data1,data2,dim=1,func=pearsonr):
    """
    Computes correlation coefficients and p-values between corresponding slices of two datasets.
    Parameters
    ----------
    data1 : AnnData or similar object
        First dataset with attribute `.X` containing the data matrix.
    data2 : AnnData or similar object
        Second dataset with attribute `.X` containing the data matrix.
    dim : int, optional (default=1)
        Dimension along which to compute correlations:
        - If 1, computes correlation for each column (e.g., gene).
        - If 0, computes correlation for each row (e.g., cell/sample).
    func : callable, optional (default=pearsonr)
        Function to compute correlation and p-value. Must accept two 1D arrays and return (correlation, p-value).
    Returns
    -------
    r1 : np.ndarray
        Array of correlation coefficients for each slice.
    p1 : np.ndarray
        Array of p-values for each slice.
    """
    
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1
