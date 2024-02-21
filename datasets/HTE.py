import numpy as np
from .base import BaseReactionDataset
from tqdm import *
import torch
from torch_geometric.data import Batch

class HTEDataset(BaseReactionDataset):
    def __init__(self, npz_file_path, modalities=['smiles']):
        """
        Initialize the dataset.
        :param npz_file_path: Path to the NPZ file containing the reaction data.
        :param modalitieHTEA list of modalities to include for each reaction.
        """
        data = np.load(npz_file_path, allow_pickle=True)
        data_df = data['data_df']
        reaction_smiles_list = [item[0] for item in data_df]
        labels = [item[1] for item in data_df]

        super().__init__(reaction_smiles_list, labels)
        self.modalities = modalities

        # Process modalities for each reaction
        self.process_modalities()

    def process_modalities(self):
        """
        Process and add additional modalities to each reaction as specified.
        """
        for reaction in tqdm(self.reactions):
            if 'smiles' in self.modalities:
                reaction.parse_reaction_smiles()
            if 'graph' in self.modalities:
                # Directly call the method to build graph representation for the reaction
                reaction.build_graph_for_reaction()


    def add_modality(self, reaction_index, modality_name, modality_data):
        """
        Add or update a modality for a specific reaction.
        :param reaction_index: Index of the reaction in the dataset.
        :param modality_name: Name of the modality to add/update.
        :param modality_data: Data for the modality.
        """
        self.reactions[reaction_index].add_modality(modality_name, modality_data)


def collate_fn(batch):
    """
    Collate function that groups the k-th reactant or product of each reaction into batches.
    
    :param batch: A list of tuples, each tuple is (reaction, yield),
                  where reaction contains lists of reactants and products as Molecule objects.
    :return: A tuple of lists, where each list contains batched graphs for the k-th reactant or product, 
             followed by a tensor of yields.
    """
    # Initialize lists to hold batched graphs for reactants and products
    reactants_batches = []
    products_batches = []
    max_reactants = max(len(reaction.reactants) for reaction, _ in batch)
    max_products = max(len(reaction.products) for reaction, _ in batch)
    
    # Initialize lists of lists for each reactant and product position
    for _ in range(max_reactants):
        reactants_batches.append([])
    for _ in range(max_products):
        products_batches.append([])
    
    yields = [yield_value for _, yield_value in batch]

    # Fill in the lists with graphs, handling varying numbers of reactants and products
    for reaction, _ in batch:
        for i, reactant in enumerate(reaction.reactants):
            reactants_batches[i].append(reactant.graph)
        for i, product in enumerate(reaction.products):
            products_batches[i].append(product.graph)
    
    # Batch all graphs in each list
    for i in range(len(reactants_batches)):
        reactants_batches[i] = Batch.from_data_list(reactants_batches[i])
    for i in range(len(products_batches)):
        products_batches[i] = Batch.from_data_list(products_batches[i])
    
    return reactants_batches, products_batches, torch.tensor(yields, dtype=torch.float)