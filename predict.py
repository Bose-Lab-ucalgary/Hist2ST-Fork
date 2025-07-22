import torch
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from dataset import ViT_HER2ST, ViT_SKIN, ViT_HEST1K
from scipy.stats import pearsonr,spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
def pk_load(fold,mode='train', sample_ids=None,flatten=False,dataset='her2st',r=4,ori=True,adj=True,prune='Grid',neighs=4):
    assert dataset in ['her2st','cscc','hest1k']
    if dataset=='her2st':
        dataset = ViT_HER2ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    elif dataset=='cscc':
        dataset = ViT_SKIN(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    elif dataset=='hest1k':
        dataset = ViT_HEST1K(
            mode=mode,fold=fold,flatten=flatten,sample_ids=sample_ids,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    return dataset
def test(model,test,device='cuda'):
    """
    Evaluates a model on test data and returns predicted and ground truth AnnData objects.
    Args:
        model: Neural network model to evaluate
        test: Test data loader containing patches, positions, expressions, and adjacency matrices
        device (str, optional): Device to run model on. Defaults to 'cuda'.
    Returns:
        tuple: Contains two AnnData objects:
            - adata: AnnData object with predicted expressions and spatial coordinates
            - adata_gt: AnnData object with ground truth expressions and spatial coordinates
    Note:
        The function runs in evaluation mode with no gradient computation.
        The spatial coordinates are stored in the 'spatial' key of obsm.
    """
    
    model=model.to(device)
    model.eval()
    
    # Get model embedding limits
    # max_x = model.x_embed.num_embeddings
    # max_y = model.y_embed.num_embeddings
    # print(f"Model expects positions in range [0, {max_x-1}] x [0, {max_y-1}]")
    
    preds=[]
    ct=[]
    gt=[]
    loss=0
    with torch.no_grad():
        for i, (patch, position, exp, adj, *_, center) in enumerate(tqdm(test)):
            print(f"Batch {i}: patch shape = {patch.shape}")
            patch, position, adj = patch.to(device), position.to(device), adj.to(device).squeeze(0)
            try:
                pred = model(patch, position, adj)[0]
            except RuntimeError as e:
                print(f"Error at model forward pass:")
                print(f"patch final shape: {patch.shape}")
                print(f"position final shape: {position.shape}")
                print(f"adj final shape: {adj.shape}")
                raise e
            # pred = model(patch, position, adj)[0]
            preds.append(pred.squeeze().cpu().numpy())
            ct.append(center.squeeze().cpu().numpy())
            gt.append(exp.squeeze().cpu().numpy())
    # Convert lists to NumPy arrays
    preds_array = np.concatenate(preds, axis=0) if len(preds) > 1 else preds[0]
    ct_array = np.concatenate(ct, axis=0) if len(ct) > 1 else ct[0]
    gt_array = np.concatenate(gt, axis=0) if len(gt) > 1 else gt[0]
    
    print(f"Final shapes - preds: {preds_array.shape}, ct: {ct_array.shape}, gt: {gt_array.shape}")
    
    # Create AnnData objects with arrays, not lists
    adata = ad.AnnData(preds_array)
    adata.obsm['spatial'] = ct_array
    
    adata_gt = ad.AnnData(gt_array)
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
