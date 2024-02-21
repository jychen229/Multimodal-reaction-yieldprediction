import numpy as np
from data import Molecule, Reaction
from torch.utils.data import Dataset, Subset

class BaseMoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels=None):
        """
        Initialize the molecule dataset.
        :param smiles_list: A list containing SMILES strings.
        :param labels: A list of labels corresponding to the molecules, optional.
        """
        self.molecules = [Molecule(smiles) for smiles in smiles_list]
        self.labels = labels if labels is not None else [None] * len(smiles_list)
    
    def __len__(self):
        """Return the number of molecules in the dataset."""
        return len(self.molecules)
    
    def __getitem__(self, idx):
        """
        Get a molecule object and its label by index.
        :param idx: Index of the molecule.
        :return: The molecule object and its label at the specified index.
        """
        return self.molecules[idx], self.labels[idx]
    
    def shuffle_data(self):
        """Shuffle the molecules and labels in the dataset."""
        indices = np.arange(len(self.molecules))
        np.random.shuffle(indices)
        self.molecules = [self.molecules[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def split_data(self, train_ratio=0.8, validate_ratio=0.1, shuffle=True):
        """
        Split the dataset into training, validation, and test subsets, with an option to shuffle.
        :param train_ratio: Proportion of the data to be used for training.
        :param validate_ratio: Proportion of the data to be used for validation.
        :param shuffle: Whether to shuffle the data.
        :return: Training, validation, and test subsets of the dataset.
        """
        if shuffle:
            self.shuffle_data()
        
        total_molecules = len(self.molecules)
        train_end = int(total_molecules * train_ratio)
        validate_end = int(total_molecules * (train_ratio + validate_ratio))
        train_dataset = Subset(self, np.arange(0, train_end))
        validate_dataset = Subset(self, np.arange(train_end, validate_end))
        test_dataset = Subset(self, np.arange(validate_end, total_molecules))
        
        return train_dataset, validate_dataset, test_dataset

class BaseReactionDataset(Dataset):
    def __init__(self, reaction_smiles_list, labels=None):
        """
        Initialize the reaction dataset.
        :param reaction_smiles_list: A list of reaction SMILES strings.
        :param labels: A list of labels corresponding to the reactions, optional.
        """
        self.reactions = [Reaction(reaction_smiles=smiles) for smiles in reaction_smiles_list]
        self.labels = labels if labels is not None else [None] * len(reaction_smiles_list)
    
    def __len__(self):
        """Return the number of reactions in the dataset."""
        return len(self.reactions)
    
    def __getitem__(self, idx):
        """
        Get a reaction object and its label by index.
        :param idx: Index of the reaction.
        :return: The reaction object and its label at the specified index.
        """
        return self.reactions[idx], self.labels[idx]
    
    def shuffle_data(self):
        """Shuffle the reactions and labels in the dataset."""
        indices = np.arange(len(self.reactions))
        np.random.shuffle(indices)
        self.reactions = [self.reactions[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def split_data(self, train_ratio=0.8, validate_ratio=0.1, shuffle=True):
        """
        Split the dataset into training, validation, and test subsets, with an option to shuffle.
        :param train_ratio: Proportion of the data to be used for training.
        :param validate_ratio: Proportion of the data to be used for validation.
        :param shuffle: Whether to shuffle the data.
        :return: Training, validation, and test subsets of the dataset.
        """
        if shuffle:
            self.shuffle_data()
        
        total_reactions = len(self.reactions)
        train_end = int(total_reactions * train_ratio)
        validate_end = int(total_reactions * (train_ratio + validate_ratio))
        
        # Create subsets for training, validation, and testing
        train_dataset = Subset(self, np.arange(0, train_end))
        validate_dataset = Subset(self, np.arange(train_end, validate_end))
        test_dataset = Subset(self, np.arange(validate_end, total_reactions))
        
        return train_dataset, validate_dataset, test_dataset
    

