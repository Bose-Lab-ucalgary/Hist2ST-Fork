import pickle
import pandas as pd

class Symbol_Converter:
    def __init__(self, to_sym_dict=None, to_ens_dict=None):
        if to_sym_dict is None:
            to_sym_path = '../data/ensembl_to_symbol.pkl'
            self.convert_to_sym = pickle.load(open(to_sym_path, 'rb'))
        else: self.convert_to_sym = to_sym_dict
        if to_ens_dict is None:
            to_ens_path = '../data/symbol_to_ensembl.pkl'
            self.convert_to_ens = pickle.load(open(to_ens_path, 'rb'))
        else: self.convert_to_ens = to_ens_dict

        dict_path = '../data/hgnc_complete_set.tsv'
        self.df = self.__load_conversion_dict(dict_path)
        
    def __load_conversion_dict(self, path):
        """Load DataFrame from TSV file"""
        df = pd.read_csv(path, sep='\t')
        return pd.Series(df.ensembl.values, index=df.symbol).to_dict()

    def convert_symbol_to_ensembl(self, symbol):
        """Convert gene symbol to Ensembl ID."""
        if symbol in self.convert_to_ens:
            return self.convert_to_ens[symbol]
    
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
        
        return ensembl
    
    def convert_symbols_to_ensembl(self, symbols, on_missing='keep'):
        """Convert a list of gene symbols to Ensembl IDs."""
        if on_missing == 'keep':
            return [self.convert_symbol_to_ensembl(symbol) if self.convert_to_ens(symbol) else symbol for symbol in symbols]
        elif on_missing == 'drop':
            return [self.convert_symbol_to_ensembl(symbol) for symbol in symbols]
        
    
    def convert_ensembl_to_symbol(self, ensembl):
        """Convert Ensembl ID to gene symbol."""
        if ensembl in self.convert_to_sym:
            return self.convert_to_sym[ensembl]
        else:
            dataframe = self.df
            symbol = dataframe[dataframe['ensembl_gene_id'] == ensembl]['symbol'].values[0]
            return symbol
        
    def convert_ensembls_to_symbols(self, ensembls, on_missing='keep'):
        """Convert a list of Ensembl IDs to gene symbols."""
        if on_missing == 'keep':
            return [self.convert_ensembl_to_symbol(ens) if ens in self.convert_to_sym else ens for ens in ensembls]
        elif on_missing == 'drop':
            return self._drop_missing_symbols(ensembls)
        return [self.convert_ensembl_to_symbol(ens) for ens in ensembls if ens in self.convert_to_sym]