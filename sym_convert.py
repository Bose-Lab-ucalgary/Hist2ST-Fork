import pickle
import pandas as pd
import numpy as np
from collections import Counter

class Symbol_Converter:
    def __init__(self, to_sym_dict=None, to_ens_dict=None):
        if to_sym_dict is None:
            to_sym_path = '../../data/ensembl_to_symbol.pkl'
            self.convert_to_sym = pickle.load(open(to_sym_path, 'rb'))
        else: self.convert_to_sym = to_sym_dict
        if to_ens_dict is None:
            to_ens_path = '../../data/symbol_to_ensembl.pkl'
            self.convert_to_ens = pickle.load(open(to_ens_path, 'rb'))
        else: self.convert_to_ens = to_ens_dict

        dict_path = '../../data/hgnc_complete_set.tsv'
        self.df = self.__load_conversion_dict(dict_path)
        
    def __load_conversion_dict(self, path):
        """Load DataFrame from TSV file"""
        df = pd.read_csv(path, sep='\t')
        return df

    def convert_symbol_to_ensembl(self, symbol, on_missing):
        """Convert gene symbol to Ensembl ID."""
        append = ''
        if '_' in symbol:
            append = symbol.split('_')[1]
            symbol = symbol.split('_')[0]
        
        if symbol in self.convert_to_ens:
            return (self.convert_to_ens[symbol] + '_' + append) if append else self.convert_to_ens[symbol]
    
        # Initialize ensembl as None
        ensembl = None
        dataframe = self.df
        
        # Try previous symbol
        prev_matches = dataframe[dataframe['previous_symbol'] == symbol]['ensembl_gene_id'].values
        if len(prev_matches) > 0:
            ensembl = prev_matches[0]
        
        # If no match in previous symbols, try alias symbols
        if ensembl is None:
            alias_matches = dataframe[dataframe['alias_symbol'] == symbol]['ensembl_gene_id'].values
            if len(alias_matches) > 0:
                ensembl = alias_matches[0]
                
        if ensembl is None:
            print(f"Error: {symbol} not in conversion dictionary.")
            if on_missing == 'keep':
                return (symbol + '_' + append) if append else symbol
            else: return None
        return (ensembl + '_' + append) if append else ensembl
    
    def convert_symbols_to_ensembl(self, symbols, on_missing='keep'):
        """Convert a list of gene symbols to Ensembl IDs."""
        return [self.convert_symbol_to_ensembl(symbol, on_missing) for symbol in symbols]


    def convert_ensembl_to_symbol(self, ensembl, on_missing):
        """Convert Ensembl ID to gene symbol."""
        append = ''
        if '_' in ensembl:
            append = ensembl.split('_')[1]
            ensembl = ensembl.split('_')[0]

        if ensembl in self.convert_to_sym:
            return (self.convert_to_sym[ensembl] + '_' + append) if append else self.convert_to_sym[ensembl]
        
        symbol = None
        dataframe = self.df
        symbol = dataframe[dataframe['ensembl_gene_id'] == ensembl]['symbol'].values
        if len(symbol) > 0:
            return (symbol[0] + '_' + append) if append else symbol[0]
        elif ensembl is None:
            print(f"Error: {ensembl} not in conversion dictionary.")
            if on_missing == 'keep':
                return (ensembl + '_' + append) if append else ensembl
        return (symbol + '_' + append) if append else symbol

    def convert_ensembls_to_symbols(self, ensembls, on_missing='keep'):
        """Convert a list of Ensembl IDs to gene symbols."""
        return [self.convert_ensembl_to_symbol(ens, on_missing) for ens in ensembls]
    
    def get_common_genes(self, base_list, list2):
        """base_list does not have duplicates, will return list2 names"""
        # Convert both lists to Python lists if they're numpy arrays
        if isinstance(base_list, np.ndarray):
            base_list = base_list.tolist()
        if isinstance(list2, np.ndarray):
            list2 = list2.tolist()
        
        # Convert base list to standard format
        converted_base_list = []
        for gene in base_list:
            if isinstance(gene, str):
                converted_base_list.append(gene)
            else:
                # Handle non-string entries
                converted_base_list.append(str(gene))
        
        # Convert second list
        converted_list2 = []
        for gene in list2:
            if isinstance(gene, str):
                converted_list2.append(gene)
            else:
                # Handle non-string entries
                converted_list2.append(str(gene))

        duplicates = [gene for gene in converted_list2 if '_1' in gene]
        clipped_duplicates = []
        if duplicates:
            converted_list2 = [gene for gene in converted_list2 if '_' not in gene]
            clipped_duplicates = [gene.split('_')[0] for gene in duplicates]
            converted_list2.extend(clipped_duplicates)
        
        # Now find common genes
        common_genes = list(set(converted_base_list) & set(converted_list2))
        
        if not common_genes:
            print("Warning: No common genes found between the two lists.")
            return [], []

        list2_names = [gene if gene in converted_list2 else (gene+"_1") for gene in common_genes]
        
        return common_genes, list2_names
    
    def get_missing_genes(self, base_list, list2):
        """Get genes in base_list that are not in list2."""
        common_genes, _ = self.get_common_genes(base_list, list2)
        missing_genes = list(set(base_list) - set(common_genes))
        if not missing_genes:
            print("Warning: No missing genes found.")
            return []
        return missing_genes