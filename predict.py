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



def test(model, test, device='cuda'):
    """
    Evaluate a model on test data and return predictions and ground truth as AnnData objects.
    Args:
        model: The neural network model to evaluate
        test: DataLoader containing test data with patches, positions, expressions, adjacency matrices, and centers
        device (str, optional): Device to run the model on. Defaults to 'cuda'.
    Returns:
        tuple: A tuple containing:
            - adata (AnnData): AnnData object with model predictions and spatial coordinates
            - adata_gt (AnnData): AnnData object with ground truth expressions and spatial coordinates
    Note:
        The function sets the model to evaluation mode and disables gradient computation.
        Only processes the last batch of the test data due to variable reassignment in the loop.
    """
    
    model=model.to(device)
    model.eval()
    preds=None
    ct=None
    gt=None
    loss=0
    with torch.no_grad():
        for patch, position, exp, adj, *_, center in tqdm(test):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device).squeeze(0)
            pred = model(patch, position, adj)[0]
            preds = pred.squeeze().cpu().numpy()
            ct = center.squeeze().cpu().numpy()
            gt = exp.squeeze().cpu().numpy()
    adata = ad.AnnData(preds)
    adata.obsm['spatial'] = ct
    adata_gt = ad.AnnData(gt)
    adata_gt.obsm['spatial'] = ct
    return adata,adata_gt

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
