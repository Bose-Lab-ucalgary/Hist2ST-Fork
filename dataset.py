import os
import glob
from turtle import pos
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
import h5py
# from utils import read_tiff, get_data, embs_to_syms, sym_to_ens
from graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from sym_convert import Symbol_Converter

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,fold=0,r=4,flatten=True,ori=False,adj=False,prune='Grid',neighs=4):
        super(ViT_HER2ST, self).__init__()
        
        self.cnt_dir = '../../data/her2st/data/ST-cnts'
        self.img_dir = '../../data/her2st/data/ST-imgs'
        self.pos_dir = '../../data/her2st/data/ST-spotfiles'
        self.lbl_dir = '../../data/her2st/data/ST-pat/lbl'
        self.r = 224//r

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.ori = ori
        self.adj = adj
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names[1:33]

        te_names = [samples[fold]]
        print(te_names)
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names} # WHY DO WE GET THE LABEL ARRAYS FOR EVERY SAMPLE??? We only need one for LOO....
            
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}
            idx=self.meta_dict[self.names[0]].index
            lbl=self.lbl_dict[self.names[0]]
            lbl=lbl.loc[idx,:]['label'].values
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]]=lbl # Only getting labels for the first sample ? Are we assuming LOO always?
        elif train:
            for i in self.names: # For each sample
                idx=self.meta_dict[i].index # get the ids for this sample
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i) # Get the labels for this sample
                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl # Assign the label in the label dictionary
                else:
                    self.label[i]=torch.full((len(idx),),-1) # If not in one of the labelled samples, assign -1 (undetermined)
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))  
            for i,m in self.meta_dict.items() # Each sample's expression, normalized and log-transformed. 
        }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
        
    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID] # pixel locations
        loc = self.loc_dict[ID] # array coordinates in "{x}x{y}" format
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label=self.label[ID]
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self,name):
        """
        Loads and processes a labeled coordinates TSV file for a given sample name.
        The method reads a TSV file containing labeled coordinates, rounds and converts
        the 'x' and 'y' columns to integers, and creates a unique 'id' for each row in the
        format '{x}x{y}'. It then sets this 'id' as the DataFrame index, and removes the
        original 'pixel_x', 'pixel_y', 'x', and 'y' columns.
        Args:
            name (str): The sample name used to locate the labeled coordinates file.
        Returns:
            pandas.DataFrame: A DataFrame indexed by the generated 'id', with the original
            coordinate columns removed.
        """
        
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df

class ViT_SKIN(torch.utils.data.Dataset):        
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,r=4,norm=False,fold=0,flatten=True,ori=False,adj=False,prune='NA',neighs=4):
        super(ViT_SKIN, self).__init__()

        self.dir = './data/GSE144240_RAW/'
        self.r = 224//r

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        gene_list = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)))
                for i,m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
                for i,m in self.meta_dict.items()
            }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID].permute(1,0,2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        adj=self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)
        
    
class ViT_HEST1K(torch.utils.data.Dataset):
    """Some Information about ViT_HEST1K (from AnnData objects)"""
    def __init__(self, r=4, norm=True, gene_list = None, sample_ids= None, mode = 'test', fold=0, flatten=True, ori=False, adj=False, prune='NA', neighs=4):
        """
        adata_dict: dict mapping sample names to AnnData objects
        gene_list: list of genes to use
        """
        super().__init__()

        self.hest_path = "/work/bose_lab/tahsin/data/HEST"
        self.norm=norm
        # self.train = train
        self.fold = fold
        self.flatten = flatten
        self.ori = ori
        self.neighs = neighs
        self.adj = adj
        self.prune = prune
        self.r = r
        self.symbol_converter = Symbol_Converter()  # Initialize symbol converter
        
        if self.ori:
            self.ori_dict = {}
            self.counts_dict = {}
            
        # TODO: if we need to match the genes to her2st?
        if mode == 'validation':
            gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
            self.gene_list = gene_list
            self.ens_gene_list = self.symbol_converter.convert_symbols_to_ensembl(gene_list, on_missing='keep')

        # Load metadata
        meta_df = pd.read_csv(os.path.join(self.hest_path, "HEST_v1_1_0.csv"))
        meta_df = meta_df[meta_df['species'] == 'Homo sapiens']

        # Split into train/test based on fold
        if sample_ids is None:
            # Use all available samples, split by fold
            all_ids = meta_df['id'].tolist()
            np.random.seed(42)
            np.random.shuffle(all_ids)
            
            # Simple 80/20 split for each fold
            split_idx = int(len(all_ids) * 0.8)
            if mode == 'train':
                self.sample_ids = all_ids[:split_idx]
            elif mode == 'test':
                self.sample_ids = all_ids[split_idx:]
        else:
            self.sample_ids = sample_ids

         # Filter existing samples
        self.sample_ids = [sid for sid in self.sample_ids 
                          if os.path.exists(os.path.join(self.hest_path, "st", f"{sid}.h5ad"))]

        print(f"HEST Dataset: {len(self.sample_ids)} samples ({'train' if mode == 'train' else 'test'})")
        print(self.sample_ids)
        
        # Load gene list (use HVG from first sample if not provided)
        if gene_list is None:
            self.gene_list = self._get_common_hvg()
        else:
            self.gene_list = gene_list
            
        print(f"Using {len(self.gene_list)} genes")
        
        # Store sample data
        self.names = self.sample_ids
        self.label = {}  # For clustering labels if needed
        
    def _get_common_hvg(self, n_genes=785):
        """Get common highly variable genes across all samples. Would be used if 
        gene_list is not provided (if we don't use the HER2ST list)"""
        all_genes = []
        for sid in self.sample_ids[:5]: #TODO: confirm limiting to just first 5 is okay
            adata_path = os.path.join(self.hest_path, "st", f"{sid}.h5ad")
            if os.path.exists(adata_path):
                adata = ad.read_h5ad(adata_path)
                sc.pp.highly_variable_genes(adata, n_top_genes=n_genes, subset=True)
                hvg = adata.var_names[adata.var['highly_variable']].index.tolist()
                all_genes.extend(hvg)
        gene_counts = pd.Series(all_genes).value_counts()
        common_genes = gene_counts.head(n_genes).index.tolist()
        return common_genes
    
    def load_sample_with_unique_indices(self, sample_id):
        """
        Load a sample with guaranteed unique indices for each spot
        
        Args:
            sample_id (str): ID of the sample to load
            
        Returns:
            tuple: (patches, positions, expression, adj, centers, spot_ids)
        """
        # Load AnnData file
        adata_path = os.path.join(self.hest_path, "st", f"{sample_id}.h5ad")
        adata = ad.read_h5ad(adata_path)
        
        # Create unique spot indices
        n_spots = len(adata.obs_names)
        spot_ids = pd.Index([f"spot_{i:05d}" for i in range(n_spots)], name='spot_id')
        
        # Update AnnData with unique indices
        adata.obs.index = spot_ids
        
        # # Process the rest of the data as normal
        if self.adj and self.ori:
            patches, positions, expression, adj_matrix, ori, centers = self.process_sample(adata, sample_id)
            return patches, positions, expression, adj_matrix, ori, centers, spot_ids
        elif self.adj:
            patches, positions, expression, adj_matrix, centers = self.process_sample(adata, sample_id)
            return patches, positions, expression, adj_matrix, centers, spot_ids
        elif self.ori:
            patches, positions, expression, ori, centers = self.process_sample(adata, sample_id)
            return patches, positions, expression, ori, centers, spot_ids
        else:
            patches, positions, expression, centers = self.process_sample(adata, sample_id)
            return patches, positions, expression, centers, spot_ids

        # return self.process_sample(adata, sample_id)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        # return self.load_sample_with_unique_indices(sample_id)
        
        adata_path = os.path.join(self.hest_path, "st", f"{sample_id}.h5ad")
        adata = ad.read_h5ad(adata_path)
        
        return self.process_sample(adata, sample_id)
        
    def process_sample(self, adata, sample_id):
        adata_path = os.path.join(self.hest_path, "st", f"{sample_id}.h5ad")
        adata = ad.read_h5ad(adata_path)
        # Inside ViT_HEST1K.__getitem__
        print(f"\nLoading sample {sample_id}")
        print(f"AnnData object:")
        print(f"  obs_names unique: {adata.obs_names.is_unique}")
        print(f"  var_names unique: {adata.var_names.is_unique}")
        
        if len(adata.var_names) == 0:
            print(f"Warning: Sample {sample_id} has no genes. Skipping.")
            return None
        # Debug print statements
        print(f"\nProcessing sample {sample_id}")
        # print(f"Gene list contains HPS6: {'HPS6' in self.gene_list}")
        # print(f"AnnData contains HPS6: {'HPS6' in adata.var_names}")
        
        # Make var_names unique before any indexing
        if not adata.var_names.is_unique:
            print(f"Found {sum(adata.var_names.duplicated())} duplicate gene names, making them unique")
            # Method 1: Make unique by appending _1, _2, etc. to duplicates
            adata.var_names_make_unique()
        
        # Create dictionaries for faster lookups
        gene_indices = {gene: idx for idx, gene in enumerate(adata.var_names)}
        
        # Initialize expression matrix with zeros (all genes from gene_list)
        n_spots = adata.shape[0]
        n_genes = len(self.gene_list)
        exps = np.zeros((n_spots, n_genes))

        common_genes, adata_common_labels = self.symbol_converter.get_common_genes(self.gene_list, adata.var_names)
        missing_genes = self.symbol_converter.get_missing_genes(self.gene_list, adata.var_names)
        
        print(f"Sample {sample_id} has {len(common_genes)} common genes with the dataset")
        
        # Convert gene lists to regular Python strings
        common_genes = [str(gene) for gene in common_genes]
        adata_common_labels = [str(label) for label in adata_common_labels]
        # missing_genes = [str(gene) for gene in missing_genes]
        # gene_list = [str(gene) for gene in self.gene_list]
        
        # Create a boolean mask for gene selection
        gene_mask = adata.var_names.isin(adata_common_labels)
        if hasattr(adata.X, "toarray"):
            exps = adata.X[:, gene_mask].toarray()
        else:
            exps = adata.X[:, gene_mask]
    
        # Verify we got the right number of genes
        if exps.shape[1] != len(common_genes):
            print(f"Warning: Expected {len(common_genes)} genes but got {exps.shape[1]}")
            print(f"First few common genes: {common_genes[:5]}")
            print(f"First few var names: {adata.var_names[:5]}")
            
        # Add zero columns for missing genes
        zero_cols = np.zeros((exps.shape[0], len(missing_genes)))
        exps = np.hstack([exps, zero_cols])
            
        # Reorder columns to match gene_list order
        # Create a dictionary mapping genes to their positions
        gene_order = {gene: idx for idx, gene in enumerate(common_genes + missing_genes)}

        # Create column order to match self.gene_list
        col_order = [gene_order[gene] for gene in self.gene_list]

        # Reorder columns
        exps = exps[:, col_order]
                
        if not adata.obs_names.is_unique:
            print("Warning: Found duplicate observation names!")        
        ori_exps = exps.copy()  # Keep original for size factors if needed
        
        #TODO: they normalized and log-transformed the data in the HER2ST dataset, should we do that here?
        norm_exps = scp.transform.log(scp.normalize.library_size_normalize(ori_exps))
            
        # Get array coordinates
        # pos = adata.obs[['array_row', 'array_col']].values.astype(int)
        if 'array_row' in adata.obs and 'array_col' in adata.obs:
            pos = adata.obs[['array_row', 'array_col']].values.astype(int)
        elif 'spatial' in adata.obsm:
            pos = adata.obsm['spatial'].copy()
        else:
            print(f"Error: Sample {sample_id} does not have 'array_row' and 'array_col' in obs or 'spatial' in obsm.")
            pos = np.zeros((adata.n_obs, 2), dtype=int)  
            for i in range(adata.n_obs):
                pos[i] = [i // 64, i % 64]  
        
        
        pos_min = pos.min(axis=0)
        pos_max = pos.max(axis=0)
        
        # Normalize positions to [0, 1] range
        pos_normalized = (pos - pos_min) / (pos_max - pos_min + 1e-8)
        # Scale to [0, 63]
        pos_scaled = (pos_normalized * 63).astype(int)
        # Ensure positions are within bounds
        pos_scaled = np.clip(pos_scaled, 0, 63)
        
        # Get pixel coordinates
        centers = adata.obsm['spatial']

        # Load Patches
        patch_path = os.path.join(self.hest_path, "patches", f"{sample_id}.h5")
        if os.path.exists(patch_path):
            patches = self._load_patches(sample_id, adata.obs_names)
            # Resize patches from 224x224 to 112x112 using interpolation
            patches = F.interpolate(torch.from_numpy(patches), size=(112, 112), mode='bilinear', align_corners=True).numpy()
        else:
            patches = torch.randn(len(adata), 3, 224, 224)
            
        # Get adjacency matrix if required
        if self.adj:
            adj_matrix = calcADJ(pos, self.neighs, pruneTag=self.prune)
        else:
            adj_matrix = None
            
        # Get original counts and size factors if requested
        if self.ori:
            ori_counts = ori_exps
            
            # Calculate size factors
            n_counts = ori_counts.sum(1)
            sf = n_counts / np.median(n_counts)
            
            # Store in dictionaries
            self.ori_dict[sample_id] = ori_counts
            self.counts_dict[sample_id] = sf    
        
        patches = torch.FloatTensor(patches)
        positions = torch.LongTensor(pos_scaled)  # Ensure positions are integers
        expression = torch.FloatTensor(exps)
        adj_matrix = torch.FloatTensor(adj_matrix) if adj_matrix is not None else None
        
        if self.adj and self.ori:
            return patches, positions, expression, adj_matrix, [torch.FloatTensor(self.ori_dict[sample_id]), torch.FloatTensor(self.counts_dict[sample_id])], centers
        elif self.adj:
            return patches, positions, expression, adj_matrix, centers
        elif self.ori:
            return patches, positions, expression, [torch.FloatTensor(self.ori_dict[sample_id]), torch.FloatTensor(self.counts_dict[sample_id])], centers
        else: return patches, positions, expression, centers


    def __len__(self):
        return len(self.sample_ids)

    def _load_patches(self, sample_id, spot_names):
        patches = []
        path = os.path.join(self.hest_path, "patches", f"{sample_id}.h5")
        
        with h5py.File(path, 'r') as f:
            images = f['img'][:]
            barcodes = [bc[0].decode('utf-8') if isinstance(bc[0], bytes) else bc[0] for bc in f['barcode'][:]]
            
            barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
            
            
            for spot in spot_names:
                if spot in barcode_to_idx:
                    idx = barcode_to_idx[spot]
                    img = images[idx]
                    # patches.append(images[idx])

                    # Why convert to tensor and normalize??
                    if len(img.shape) == 2:
                        img = np.stack([img, img, img], axis =0) # Convert grayscale to RGB
                    else:
                        img = img.transpose(2, 0, 1) # Convert HxWxC to CxHxW
                        
                    patches.append(img)
                    
                else:
                    patches.append(np.zeros((3, 224, 224)))
                    
        return np.array(patches)