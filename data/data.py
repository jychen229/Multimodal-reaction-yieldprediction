from rdkit import Chem
from torch_geometric.data import Data as PyGData
import dgl
from utils import smiles_to_graph_features, graph_2d_features

class Molecule:
    def __init__(self, smiles: str, graph = None, **attr):
        self.smiles = smiles
        self.features = {}  
        self.graph = graph   
        self.modalities = {}
        self.tokenized_smiles = None

    def add_feature(self, name, value):
        """add or update molecular features"""
        self.features[name] = value

    def get_feature(self, name):
        return self.features.get(name, None)

    def add_modality(self, modality_name: str, modality_data):
        self.modalities[modality_name] = modality_data

    def get_modality(self, modality_name):
        return self.modalities.get(modality_name, None)

    def build_2d_graph(self, x, edge_index, edge_attr, pkg='pyg'):
        if pkg == 'pyg':
            self.graph = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr)
        elif pkg == 'dgl':
            src, dst = edge_index
            self.graph = dgl.graph((src, dst), num_nodes=x.size(0))
            self.graph.ndata['x'] = x
            self.graph.edata['edge_attr'] = edge_attr
        else:
            raise ValueError("Unsupported graph library. Choose 'PyG' or 'DGL'.")
        
    def tokenize_reaction_smiles(self, tokenizer):
        """
        Tokenize the SMILES string using the provided tokenizer.
        :param tokenizer: A tokenizer object with a `tokenize` method that accepts a SMILES string.
        """
        if not hasattr(tokenizer, "tokenize"):
            raise ValueError("The provided tokenizer does not have a 'tokenize' method.")
        self.tokenized_smiles = tokenizer.tokenize(self.smiles)

class Reaction:
    def __init__(self, reactants = None, products = None, conditions=None, reaction_smiles=None, reaction_yield=None, tokenized_smiles=None):
        self.reactants = reactants if reactants else []
        self.products = products if products else []
        self.conditions = conditions if conditions else {}  # Dictionary for reaction conditions
        self.reaction_smiles = reaction_smiles  # Original reaction SMILES string
        self.reaction_yield = reaction_yield  # Reaction yield as a percentage
        self.tokenized_smiles = tokenized_smiles  # Tokenized version of the reaction SMILES string

    def add_reactant(self, reactant: Molecule):
        """Add a reactant to the reaction."""
        self.reactants.append(reactant)

    def add_product(self, product: Molecule):
        """Add a product to the reaction."""
        self.products.append(product)

    def set_condition(self, condition_name: str, condition_value):
        """Set a condition for the reaction."""
        self.conditions[condition_name] = condition_value

    def parse_reaction_smiles(self):
        """Parse reactants and products from the reaction SMILES string."""
        if self.reaction_smiles:
            parts = self.reaction_smiles.split(">>")
            if len(parts) == 2:
                reactant_smiles, product_smiles = parts
                # Splitting each part by "." to handle multiple molecules
                reactant_smiles_list = reactant_smiles.split(".")
                product_smiles_list = product_smiles.split(".")
                # Creating Molecule instances for reactants and products
                self.reactants = [Molecule(smiles=r) for r in reactant_smiles_list]
                self.products = [Molecule(smiles=p) for p in product_smiles_list]
            else:
                raise ValueError("Invalid reaction SMILES format. Expected 'reactants >> products'.")
        else:
            raise ValueError("Reaction SMILES string is not provided.")

    def get_reaction_smiles(self):
        """Return the reaction SMILES string."""
        if self.reaction_smiles:
            return self.reaction_smiles
        else:
            raise ValueError("Reaction SMILES string is not provided.")
        
    def tokenize_reaction_smiles(self, tokenizer):
        """
        Tokenize the reaction SMILES string using the provided tokenizer.
        
        :param tokenizer: A tokenizer object with a `tokenize` method that accepts a SMILES string.
        """
        if not self.reaction_smiles:
            raise ValueError("Reaction SMILES string is not provided.")
        if not hasattr(tokenizer, "tokenize"):
            raise ValueError("The provided tokenizer does not have a 'tokenize' method.")
        
        self.tokenized_smiles = tokenizer.tokenize(self.reaction_smiles)

    def build_graph_for_reaction(self, pkg = 'pyg'):
        """
        Build graph representations for reactants and products in the reaction.
        """
        for reactant in self.reactants:
            x, edge_index, edge_attr = smiles_to_graph_features(reactant.smiles, graph_2d_features)
            reactant.build_2d_graph(x, edge_index, edge_attr, pkg=pkg)
        
        for product in self.products:
            x, edge_index, edge_attr = smiles_to_graph_features(product.smiles, graph_2d_features)
            product.build_2d_graph(x, edge_index, edge_attr, pkg=pkg)
