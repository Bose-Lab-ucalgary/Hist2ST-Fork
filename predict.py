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
def pk_load(fold,mode='train', flatten=False,dataset='her2st',r=4,ori=True,adj=True,prune='Grid',neighs=4):
    assert dataset in ['her2st', 'cscc', 'hest1k']
    
    # Debug dataset parameters
    print(f"\nLoading dataset with parameters:")
    print(f"  Dataset: {dataset}")
    print(f"  Mode: {mode}")
    print(f"  Fold: {fold}")
    
    if dataset == 'hest1k':
        dataset = ViT_HEST1K(
            mode=mode, 
            fold=fold,
            flatten=flatten,
            ori=ori,
            neighs=neighs,
            adj=adj,
            prune=prune,
            r=r
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
def test(model, test, device='cuda'):
    model = model.to(device)
    model.eval()
    
    preds = []
    ct = []
    gt = []
    
    # Debug: Print dataloader size
    print(f"Number of batches in dataloader: {len(test)}")
    
    with torch.no_grad():
        for i, (patch, position, exp, *adj_ori, center) in enumerate(tqdm(test)):
            print(f"Batch {i}: patch shape = {patch.shape}")
            if len(adj_ori) > 0:
                adj = adj_ori[0]
            else:
                adj = None
            patch, position, adj = patch.to(device), position.to(device), adj.to(device).squeeze(0)
            try:
                pred = model(patch, position, adj)[0]
                preds.append(pred.squeeze().cpu().numpy())
                ct.append(center.squeeze().cpu().numpy())
                gt.append(exp.squeeze().cpu().numpy())
            except RuntimeError as e:
                print(f"Error at model forward pass:")
                print(f"patch final shape: {patch.shape}")
                print(f"position final shape: {position.shape}")
                print(f"adj final shape: {adj.shape}")
                raise e

    if len(preds) == 0:
        raise ValueError("No predictions collected - check if dataloader is empty")

    # Convert lists to arrays
    preds_array = np.concatenate(preds, axis=0) if len(preds) > 1 else preds[0]
    ct_array = np.concatenate(ct, axis=0) if len(ct) > 1 else ct[0]
    gt_array = np.concatenate(gt, axis=0) if len(gt) > 1 else gt[0]
    
    # print(f"Final shapes - preds: {preds_array.shape}, ct: {ct_array.shape}, gt: {gt_array.shape}")

    # Create unique indices for AnnData objects
    n_spots = preds_array.shape[0]
    spot_ids = [f"spot_{i}" for i in range(n_spots)]
    
    # Create AnnData objects with explicit indices
    # adata = ad.AnnData(
    #     X=preds_array,
    #     obs=pd.DataFrame(index=spot_ids),
    #     dtype=np.float32
    # )
    # adata.obsm['spatial'] = ct_array
    
    # adata_gt = ad.AnnData(
    #     X=gt_array,
    #     obs=pd.DataFrame(index=pd.Index(spot_ids, name='spot_id').copy()),
    #     dtype=np.float32
    # )
    # adata_gt.obsm['spatial'] = ct_array
    
    # return adata, adata_gt
    # Extract spatial coordinates if they exist
    if len(ct_array.shape) == 2 and ct_array.shape[1] >= 2:
        array_row = ct_array[:, 0]
        array_col = ct_array[:, 1]
        
        # Create observation DataFrame with required columns
        obs_df = pd.DataFrame({
            'array_row': array_row,
            'array_col': array_col
        }, index=spot_ids)
    else:
        # Just create index without spatial coords
        obs_df = pd.DataFrame(index=spot_ids)
    
    # Create AnnData objects with proper observation dataframes
    try:
        adata = ad.AnnData(
            X=preds_array,
            obs=obs_df,
            dtype=np.float32
        )
        
        # Store the full spatial coordinates in obsm
        adata.obsm['spatial'] = ct_array
        
        # Create separate obs_df for ground truth to avoid shared references
        obs_df_gt = obs_df.copy()
        
        adata_gt = ad.AnnData(
            X=gt_array,
            obs=obs_df_gt,
            dtype=np.float32
        )
        adata_gt.obsm['spatial'] = ct_array
        
        # Verify indices are unique
        print(f"Pred index is unique: {adata.obs.index.is_unique}")
        print(f"GT index is unique: {adata_gt.obs.index.is_unique}")
        
    except Exception as e:
        print(f"Error creating AnnData objects: {str(e)}")
        # Create minimal AnnData as fallback
        adata = ad.AnnData(X=preds_array, dtype=np.float32)
        adata_gt = ad.AnnData(X=gt_array, dtype=np.float32)
        # Add spatial info to obsm directly
        adata.obsm['spatial'] = ct_array
        adata_gt.obsm['spatial'] = ct_array
    
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
    def get_R(data1, data2, dim=1, func=pearsonr):
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
