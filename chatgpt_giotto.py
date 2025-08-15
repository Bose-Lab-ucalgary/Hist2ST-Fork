
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import networkx as nx
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class SpatialTranscriptomics:
    """
    Python implementation of Giotto-like spatial transcriptomics analysis
    """
    
    def __init__(self, expression_matrix: np.ndarray, coordinates: np.ndarray, 
                 gene_names: Optional[List[str]] = None, spot_names: Optional[List[str]] = None):
        """
        Initialize spatial transcriptomics object
        
        Args:
            expression_matrix: (n_spots, n_genes) expression matrix
            coordinates: (n_spots, 2) spatial coordinates [x, y]
            gene_names: List of gene names
            spot_names: List of spot/cell names
        """
        self.expression = expression_matrix
        self.coordinates = coordinates
        self.n_spots, self.n_genes = expression_matrix.shape
        
        self.gene_names = gene_names or [f"Gene_{i}" for i in range(self.n_genes)]
        self.spot_names = spot_names or [f"Spot_{i}" for i in range(self.n_spots)]
        
        # Computed attributes
        self.spatial_network = None
        self.hvgs = None
        self.spatial_genes = None
        self.clusters = None
        
    def create_spatial_network(self, method='knn', k=6, distance_threshold=None):
        """
        Create spatial network between spots
        
        Args:
            method: 'knn' or 'distance'
            k: number of nearest neighbors (for knn)
            distance_threshold: maximum distance for connections (for distance)
        """
        if method == 'knn':
            nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.coordinates)
            distances, indices = nbrs.kneighbors(self.coordinates)
            
            # Create adjacency matrix
            adj_matrix = np.zeros((self.n_spots, self.n_spots))
            for i in range(self.n_spots):
                for j in indices[i][1:]:  # Skip self (first index)
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
                    
        elif method == 'distance':
            if distance_threshold is None:
                distance_threshold = np.mean(pdist(self.coordinates))
            
            distances = squareform(pdist(self.coordinates))
            adj_matrix = (distances <= distance_threshold).astype(int)
            np.fill_diagonal(adj_matrix, 0)
        
        self.spatial_network = adj_matrix
        return adj_matrix
    
    def find_highly_variable_genes(self, n_top_genes=2000, flavor='seurat'):
        """
        Identify highly variable genes
        """
        if flavor == 'seurat':
            # Calculate mean and variance for each gene
            means = np.mean(self.expression, axis=0)
            variances = np.var(self.expression, axis=0)
            
            # Calculate coefficient of variation
            cv = np.sqrt(variances) / (means + 1e-12)
            
            # Select top variable genes
            top_indices = np.argsort(cv)[::-1][:n_top_genes]
            
        elif flavor == 'cell_ranger':
            # Use normalized variance
            means = np.mean(self.expression, axis=0)
            variances = np.var(self.expression, axis=0)
            
            # Fit curve and calculate residuals
            from scipy import interpolate
            valid_mask = means > 0
            f = interpolate.interp1d(np.log10(means[valid_mask] + 1), 
                                   np.log10(variances[valid_mask] + 1), 
                                   kind='linear', fill_value='extrapolate')
            expected_var = 10 ** f(np.log10(means + 1))
            residuals = np.log10(variances + 1) - np.log10(expected_var + 1)
            
            top_indices = np.argsort(residuals)[::-1][:n_top_genes]
        
        self.hvgs = top_indices
        return top_indices, [self.gene_names[i] for i in top_indices]
    
    def calculate_spatial_autocorrelation(self, genes=None, method='morans_i'):
        """
        Calculate spatial autocorrelation (Moran's I or Geary's C)
        """
        if self.spatial_network is None:
            self.create_spatial_network()
        
        if genes is None:
            genes = range(self.n_genes)
        elif isinstance(genes, list) and isinstance(genes[0], str):
            genes = [self.gene_names.index(g) for g in genes]
        
        autocorr_scores = []
        p_values = []
        
        for gene_idx in genes:
            gene_expr = self.expression[:, gene_idx]
            
            if method == 'morans_i':
                score, p_val = self._morans_i(gene_expr, self.spatial_network)
            elif method == 'gearys_c':
                score, p_val = self._gearys_c(gene_expr, self.spatial_network)
            
            autocorr_scores.append(score)
            p_values.append(p_val)
        
        return np.array(autocorr_scores), np.array(p_values)
    
    def _morans_i(self, gene_expr, weights):
        """Calculate Moran's I statistic"""
        n = len(gene_expr)
        mean_expr = np.mean(gene_expr)
        
        # Calculate Moran's I
        numerator = np.sum(weights * np.outer(gene_expr - mean_expr, gene_expr - mean_expr))
        denominator = np.sum(weights) * np.sum((gene_expr - mean_expr) ** 2)
        
        if denominator == 0:
            return 0, 1
        
        morans_i = (n / np.sum(weights)) * (numerator / denominator)
        
        # Simple p-value approximation (for proper calculation, use permutation test)
        expected_i = -1 / (n - 1)
        variance_i = (n * n - 3 * n + 3) / ((n - 1) * (n - 2) * (n - 3))
        z_score = (morans_i - expected_i) / np.sqrt(variance_i)
        p_value = 2 * (1 - abs(z_score))  # Simplified
        
        return morans_i, p_value
    
    def _gearys_c(self, gene_expr, weights):
        """Calculate Geary's C statistic"""
        n = len(gene_expr)
        
        numerator = np.sum(weights * (gene_expr[:, np.newaxis] - gene_expr) ** 2)
        denominator = 2 * np.sum(weights) * np.var(gene_expr)
        
        if denominator == 0:
            return 1, 1
        
        gearys_c = (n - 1) * numerator / denominator
        
        # Simplified p-value
        expected_c = 1
        p_value = abs(gearys_c - expected_c)  # Simplified
        
        return gearys_c, p_value
    
    def identify_spatial_genes(self, method='morans_i', p_threshold=0.05, 
                             score_threshold=0.1, n_top=None):
        """
        Identify spatially variable genes
        """
        scores, p_values = self.calculate_spatial_autocorrelation(method=method)
        
        # Filter by significance and score
        significant_mask = p_values < p_threshold
        score_mask = scores > score_threshold
        spatial_mask = significant_mask & score_mask
        
        spatial_gene_indices = np.where(spatial_mask)[0]
        
        if n_top is not None:
            # Select top N by score
            sorted_indices = np.argsort(scores[spatial_gene_indices])[::-1][:n_top]
            spatial_gene_indices = spatial_gene_indices[sorted_indices]
        
        self.spatial_genes = spatial_gene_indices
        return spatial_gene_indices, [self.gene_names[i] for i in spatial_gene_indices]
    
    def spatial_clustering(self, method='leiden', resolution=0.5, n_clusters=None):
        """
        Perform spatial clustering
        """
        if self.spatial_network is None:
            self.create_spatial_network()
        
        if method == 'leiden' or method == 'louvain':
            # Use networkx for graph-based clustering
            G = nx.from_numpy_array(self.spatial_network)
            
            # Simple community detection (placeholder for proper Leiden/Louvain)
            from sklearn.cluster import SpectralClustering
            clustering = SpectralClustering(n_clusters=n_clusters or 8, 
                                          affinity='precomputed', 
                                          random_state=42)
            clusters = clustering.fit_predict(self.spatial_network)
            
        elif method == 'kmeans':
            # K-means on expression + spatial coordinates
            features = np.hstack([self.expression, self.coordinates])
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=n_clusters or 8, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
        elif method == 'dbscan':
            # DBSCAN on spatial coordinates
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(self.coordinates)
        
        self.clusters = clusters
        return clusters
    
    def gene_coexpression_network(self, genes=None, correlation_threshold=0.7, 
                                method='pearson'):
        """
        Build gene co-expression network
        """
        if genes is None:
            if self.hvgs is not None:
                gene_indices = self.hvgs
            else:
                gene_indices = range(min(1000, self.n_genes))  # Top 1000 genes
        else:
            gene_indices = genes
        
        # Calculate correlation matrix
        expr_subset = self.expression[:, gene_indices]
        
        if method == 'pearson':
            corr_matrix = np.corrcoef(expr_subset.T)
        elif method == 'spearman':
            corr_matrix = np.zeros((len(gene_indices), len(gene_indices)))
            for i in range(len(gene_indices)):
                for j in range(len(gene_indices)):
                    corr_matrix[i, j] = spearmanr(expr_subset[:, i], expr_subset[:, j])[0]
        
        # Create network from correlations
        adj_matrix = (np.abs(corr_matrix) > correlation_threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        
        return adj_matrix, corr_matrix
    
    def visualize_spatial_expression(self, genes, figsize=(15, 5), cmap='viridis'):
        """
        Visualize spatial expression patterns
        """
        if isinstance(genes, str):
            genes = [genes]
        
        n_genes = len(genes)
        fig, axes = plt.subplots(1, n_genes, figsize=(figsize[0], figsize[1]))
        if n_genes == 1:
            axes = [axes]
        
        for i, gene in enumerate(genes):
            if isinstance(gene, str):
                gene_idx = self.gene_names.index(gene)
            else:
                gene_idx = gene
                gene = self.gene_names[gene_idx]
            
            expr = self.expression[:, gene_idx]
            scatter = axes[i].scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                                    c=expr, cmap=cmap, s=20)
            axes[i].set_title(f'{gene}')
            axes[i].set_xlabel('Spatial X')
            axes[i].set_ylabel('Spatial Y')
            plt.colorbar(scatter, ax=axes[i])
        
        plt.tight_layout()
        return fig
    
    def visualize_clusters(self, figsize=(8, 6)):
        """
        Visualize spatial clusters
        """
        if self.clusters is None:
            self.spatial_clustering()
        
        plt.figure(figsize=figsize)
        scatter = plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], 
                            c=self.clusters, cmap='tab20', s=20)
        plt.title('Spatial Clusters')
        plt.xlabel('Spatial X')
        plt.ylabel('Spatial Y')
        plt.colorbar(scatter)
        return plt.gcf()
    
    def differential_expression_analysis(self, cluster1, cluster2=None):
        """
        Perform differential expression analysis between clusters
        """
        if self.clusters is None:
            raise ValueError("Run spatial_clustering first")
        
        if cluster2 is None:
            # One vs all
            mask1 = self.clusters == cluster1
            mask2 = self.clusters != cluster1
        else:
            # Cluster vs cluster
            mask1 = self.clusters == cluster1
            mask2 = self.clusters == cluster2
        
        expr1 = self.expression[mask1]
        expr2 = self.expression[mask2]
        
        # Calculate fold changes and p-values
        mean1 = np.mean(expr1, axis=0)
        mean2 = np.mean(expr2, axis=0)
        
        fold_changes = np.log2((mean1 + 1e-12) / (mean2 + 1e-12))
        
        # Simple t-test (for proper analysis, use more sophisticated methods)
        from scipy.stats import ttest_ind
        p_values = []
        for i in range(self.n_genes):
            _, p_val = ttest_ind(expr1[:, i], expr2[:, i])
            p_values.append(p_val)
        
        return fold_changes, np.array(p_values)
    
    def enrichment_analysis(self, gene_list, background=None):
        """
        Simple gene set enrichment analysis
        """
        # This is a placeholder - in practice, you'd use proper GO/pathway databases
        if background is None:
            background = self.gene_names
        
        # Calculate enrichment score (simplified)
        overlap = len(set(gene_list) & set(background))
        enrichment_score = overlap / len(gene_list)
        
        return {
            'enrichment_score': enrichment_score,
            'overlap_genes': list(set(gene_list) & set(background)),
            'p_value': 0.05  # Placeholder
        }

# Helper function to create SpatialTranscriptomics object from predictions
def create_spatial_object_from_predictions(pred, gt, coordinates=None, gene_names=None):
    """
    Create SpatialTranscriptomics object from model predictions
    """
    # Convert to numpy
    if hasattr(pred, 'cpu'):
        pred = pred.cpu().numpy()
    if hasattr(gt, 'cpu'):
        gt = gt.cpu().numpy()
    
    # If no coordinates provided, create a grid
    if coordinates is None:
        n_spots = pred.shape[0]
        grid_size = int(np.sqrt(n_spots))
        x_coords = np.tile(np.arange(grid_size), grid_size)[:n_spots]
        y_coords = np.repeat(np.arange(grid_size), grid_size)[:n_spots]
        coordinates = np.column_stack([x_coords, y_coords])
    
    return SpatialTranscriptomics(pred, coordinates, gene_names)

# Example usage with your predictions
def analyze_spatial_predictions(pred, gt, coordinates=None, gene_names=None):
    """
    Complete spatial analysis pipeline
    """
    # Create spatial objects
    pred_spatial = create_spatial_object_from_predictions(pred, gt, coordinates, gene_names)
    gt_spatial = create_spatial_object_from_predictions(gt, gt, coordinates, gene_names)
    
    print("=== SPATIAL TRANSCRIPTOMICS ANALYSIS ===")
    
    # 1. Find highly variable genes
    print("\n1. Finding highly variable genes...")
    hvg_indices, hvg_names = pred_spatial.find_highly_variable_genes(n_top_genes=500)
    print(f"Found {len(hvg_indices)} highly variable genes")
    
    # 2. Identify spatial genes
    print("\n2. Identifying spatially variable genes...")
    pred_spatial.create_spatial_network(method='knn', k=6)
    spatial_indices, spatial_names = pred_spatial.identify_spatial_genes(n_top=100)
    print(f"Found {len(spatial_indices)} spatial genes")
    
    # 3. Spatial clustering
    print("\n3. Performing spatial clustering...")
    clusters = pred_spatial.spatial_clustering(method='kmeans', n_clusters=8)
    print(f"Identified {len(np.unique(clusters))} clusters")
    
    # 4. Gene co-expression network
    print("\n4. Building gene co-expression network...")
    coexpr_adj, coexpr_corr = pred_spatial.gene_coexpression_network(
        genes=hvg_indices[:200], correlation_threshold=0.7)
    print(f"Co-expression network has {np.sum(coexpr_adj)/2:.0f} edges")
    
    # 5. Visualizations
    print("\n5. Creating visualizations...")
    
    # Plot top spatial genes
    if len(spatial_indices) > 0:
        top_spatial = spatial_indices[:min(3, len(spatial_indices))]
        fig1 = pred_spatial.visualize_spatial_expression(top_spatial)
        plt.savefig('spatial_genes_expression.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot clusters
    fig2 = pred_spatial.visualize_clusters()
    plt.savefig('spatial_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'pred_spatial': pred_spatial,
        'gt_spatial': gt_spatial,
        'hvg_genes': hvg_names,
        'spatial_genes': spatial_names,
        'clusters': clusters,
        'coexpression_network': (coexpr_adj, coexpr_corr)
    }