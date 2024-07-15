import numpy as np
import pandas as pd
from rdkit import Chem
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from dgl import DGLGraph
from rdkit.Chem import rdMolDescriptors as rdDesc
from data import rxn
from tqdm import tqdm


class data_process():
    '''
    return: Samples with molecular graphs (dgl.graph); smiles (list); features (np.array); yields (np.array)
    '''
    def __init__(self, folder_path, graph = True, smiles = True, features = True):
        self.use_graph = graph
        self.use_smile = smiles
        self.use_features = features
        self.folder = folder_path
        self.mols_folder = folder_path+'/molecules'
        self.rxns = rxn(self.folder)
        self.x_map = { 'atomic_num':list(range(0, 40)),'chirality': [
                            'CHI_UNSPECIFIED',
                            'CHI_TETRAHEDRAL_CW',
                            'CHI_TETRAHEDRAL_CCW',
                            'CHI_OTHER',
                            'CHI_TETRAHEDRAL',
                            'CHI_ALLENE',
                            'CHI_SQUAREPLANAR',
                            'CHI_TRIGONALBIPYRAMIDAL',
                            'CHI_OCTAHEDRAL',
                        ],'degree':list(range(0, 11)),
                        'formal_charge':list(range(-5, 7)),
                        'num_hs':list(range(0, 9)),
                        'num_radical_electrons':list(range(0, 5)),
                        'hybridization': ['UNSPECIFIED','S','SP','SP2','SP3',
                            'SP3D',
                            'SP3D2',
                            'OTHER',
                        ], 'is_aromatic': [False, True],'is_in_ring': [False, True] }
        self.e_map = { 'bond_type': [
                        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE','ONEANDAHALF',
                        'TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF','AROMATIC','IONIC','HYDROGEN',
                        'THREECENTER','DATIVEONE','DATIVE','DATIVEL','DATIVER','OTHER','ZERO',
                    ],
                    'stereo': [
                        'STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS',
                    ],
                    'is_conjugated': [False, True],
                    }



    def get_labels(self, ids):
        Y = []
        for id in ids:
            y = self.rxns.get_yield(id)
            Y.append(y)
        return np.array(Y)     


    def get_raw_feature(self, ids):
        X = []
        X_fp, X_aimnet, X_mordred, X_aev = [], [], [], []

        for id in ids:
            context = self.rxns.get_context_one_hot(id)
            fp = self.rxns.get_fp(id)
            context_fp0 = list(np.array(context + fp)) # 3807
            X_fp.append(context_fp0)
            
            t, T = self.rxns.get_tT(id)
            tT0 = [(t-14.20)/14.38, (T-20.93)/11.13]  #standardize 2
            aimnet0 = self.rxns.get_aimnet_descriptors_(id)  #51
            aimnet = aimnet0 + tT0   
            X_aimnet.append(aimnet)
            
            mordred0 = self.rxns.get_mordred(id)  #4944
            X_mordred.append(mordred0)
            
            aev0 = self.rxns.get_aev_(id) # 5148
            X_aev.append(aev0)
            x = aimnet0 + context_fp0 + mordred0 + aev0 + tT0
            X.append(x)
            
        return np.array(X)


    def get_atom_features(self, atom):
        possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'H' , 'S','Si'] #DU代表其他原子

        x = []
        x.append(possible_atom.index(atom.GetSymbol()))
        x.append([-1,0,1,2,3,4,5,6,7,].index(atom.GetImplicitValence()))
        x.append(self.x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(self.x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(self.x_map['degree'].index(atom.GetTotalDegree()))
        x.append(self.x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(self.x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(self.x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        x.append(self.x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(self.x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(self.x_map['is_in_ring'].index(atom.IsInRing()))

        return np.array(x) 

    def get_bond_features(self,bond):

        e = []
        e.append(self.e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(self.e_map['stereo'].index(str(bond.GetStereo())))
        e.append(self.e_map['is_conjugated'].index(bond.GetIsConjugated()))

        return np.array(e) 


    def from_smile2graph(self, molecule_smiles):
        G = DGLGraph()
        molecule = Chem.MolFromSmiles(molecule_smiles)
        G.add_nodes(molecule.GetNumAtoms())
        node_features = []
        edge_features = []

        for i in range(molecule.GetNumAtoms()):
            atom_i = molecule.GetAtomWithIdx(i) 
            atom_i_features = self.get_atom_features(atom_i) 
            node_features.append(atom_i_features)

            for j in range(molecule.GetNumAtoms()):
                bond_ij = molecule.GetBondBetweenAtoms(i, j)
                if bond_ij is not None:
                    G.add_edges(i,j) 
                    bond_features_ij = self.get_bond_features(bond_ij) 
                    edge_features.append(bond_features_ij)

        G.ndata['attr'] = torch.from_numpy(np.array(node_features)).type(torch.float32)  #dgl添加原子/节点特征
        G.edata['edge_attr'] = torch.from_numpy(np.array(edge_features)).type(torch.float32) #dgl添加键/边特征
        return G

    def get_2dgraph(self, ids):
        X = []
        for id in tqdm(ids):
            tmp = self.rxns.get_smile(id)
            rmols_1 = self.from_smile2graph(tmp[0])
            rmols_2 = self.from_smile2graph(tmp[1])
            x = [rmols_1, rmols_2, self.from_smile2graph(tmp[-1])]
            X.append(x)
        return X
    
    def get_smiles(self, ids):
        X = []
        for id in ids:
            r1, r2, p1 = self.rxns.get_smile(id)
            reaction_smiles = [r1, r2, p1]
            X.append(reaction_smiles)
        return X
    
    def load_data(self, ids):
        samples = {}
        labels = self.get_labels(ids)
        features = self.get_raw_feature(ids)
        graphs = self.get_2dgraph(ids)
        smiles = self.get_smiles
        samples['yields'] = labels
        samples['graphs'] = graphs
        samples['smiles'] = smiles
        samples['features'] = features
        return samples


    
    
class my_ds(Dataset):
    def __init__(self, X_f, X_g, X_s, Y):
        """
        X:  feature graph smiles
        Y: yields
        """
        self.X_f = X_f
        self.X_g = X_g
        self.X_s = X_s
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        g1 = self.X_g[idx][0]
        g2 = self.X_g[idx][1]
        g3 = self.X_g[idx][2]
        x = self.X_f[idx][:]
        s = self.X_s[idx]
        return  (g1, g2, g3, x, s, self.Y[idx])
    
    
