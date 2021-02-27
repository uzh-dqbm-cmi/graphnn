from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
import os
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from itertools import repeat, product, chain
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict

import torch
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions


# from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def smiles_to_mol(smiles):
    return AllChem.MolFromSmiles(smiles)

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


# # https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py
# MAX_ATOM = 400
# MAX_BOND = MAX_ATOM * 2

# # https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/chemutils.py

# def get_mol(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None: 
#         return None
#     Chem.Kekulize(mol)
#     return mol

# ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
# ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
# BOND_FDIM = 5 + 6
# MAX_NB = 6
# ### basic setting from https://github.com/wengong-jin/iclr19-graph2graph/blob/master/fast_jtnn/mpn.py

# def onek_encoding_unk(x, allowable_set):
#     if x not in allowable_set:
#         x = allowable_set[-1]
#     return list(map(lambda s: x == s, allowable_set))

# def atom_features(atom):
#     return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
#             + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
#             + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
#             + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
#             + [atom.GetIsAromatic()])


# def bond_features(bond):
#     bt = bond.GetBondType()
#     stereo = int(bond.GetStereo())
#     fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
#     fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
#     return torch.Tensor(fbond + fstereo)


# # https://github.com/kexinhuang12345/DeepPurpose/blob/master/DeepPurpose/utils.py#L206
# def smiles2mpnnfeature(smiles):
# 	## mpn.py::tensorize  
# 	'''
# 		data-flow:   
# 			data_process(): apply(smiles2mpnnfeature)
# 			DBTA: train(): data.DataLoader(data_process_loader())
# 			mpnn_collate_func()
# 	## utils.smiles2mpnnfeature -> utils.mpnn_collate_func -> utils.mpnn_feature_collate_func -> encoders.MPNN.forward
# 	'''
# 	try: 
# 		padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
# 		fatoms, fbonds = [], [padding] 
# 		in_bonds,all_bonds = [], [(-1,-1)] 
# 		mol = get_mol(smiles)
# 		n_atoms = mol.GetNumAtoms()
# 		for atom in mol.GetAtoms():
# 			fatoms.append( atom_features(atom))
# 			in_bonds.append([])

# 		for bond in mol.GetBonds():
# 			a1 = bond.GetBeginAtom()
# 			a2 = bond.GetEndAtom()
# 			x = a1.GetIdx() 
# 			y = a2.GetIdx()

# 			b = len(all_bonds)
# 			all_bonds.append((x,y))
# 			fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
# 			in_bonds[y].append(b)

# 			b = len(all_bonds)
# 			all_bonds.append((y,x))
# 			fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
# 			in_bonds[x].append(b)

# 		total_bonds = len(all_bonds)
# 		fatoms = torch.stack(fatoms, 0) 
# 		fbonds = torch.stack(fbonds, 0) 
# 		agraph = torch.zeros(n_atoms,MAX_NB).long()
# 		bgraph = torch.zeros(total_bonds,MAX_NB).long()
# 		for a in range(n_atoms):
# 			for i,b in enumerate(in_bonds[a]):
# 				agraph[a,i] = b

# 		for b1 in range(1, total_bonds):
# 			x,y = all_bonds[b1]
# 			for i,b2 in enumerate(in_bonds[x]):
# 				if all_bonds[b2][0] != y:
# 					bgraph[b1,i] = b2

# 	except: 
# 		print('Molecules not found and change to zero vectors..')
# 		fatoms = torch.zeros(0,39)
# 		fbonds = torch.zeros(0,50)
# 		agraph = torch.zeros(0,6)
# 		bgraph = torch.zeros(0,6)
# 	#fatoms, fbonds, agraph, bgraph = [], [], [], [] 
# 	#print(fatoms.shape, fbonds.shape, agraph.shape, bgraph.shape)
# 	Natom, Nbond = fatoms.shape[0], fbonds.shape[0]   

# 	''' 
# 	## completion to make feature size equal. 
# 	MAX_ATOM = 100
# 	MAX_BOND = 200
# 	'''
# 	atoms_completion_num = MAX_ATOM - fatoms.shape[0]
# 	bonds_completion_num = MAX_BOND - fbonds.shape[0]
# 	try:
# 		assert atoms_completion_num >= 0 and bonds_completion_num >= 0
# 	except:
# 		raise Exception("Please increasing MAX_ATOM in line 26 utils.py, for example, MAX_ATOM=600 and reinstall it via 'python setup.py install'. The current setting is for small molecule. ")


# 	fatoms_dim = fatoms.shape[1]
# 	fbonds_dim = fbonds.shape[1]
# 	fatoms = torch.cat([fatoms, torch.zeros(atoms_completion_num, fatoms_dim)], 0)
# 	fbonds = torch.cat([fbonds, torch.zeros(bonds_completion_num, fbonds_dim)], 0)
# 	agraph = torch.cat([agraph.float(), torch.zeros(atoms_completion_num, MAX_NB)], 0)
# 	bgraph = torch.cat([bgraph.float(), torch.zeros(bonds_completion_num, MAX_NB)], 0)
# 	# print("atom size", fatoms.shape[0], agraph.shape[0])
# 	# print("bond size", fbonds.shape[0], bgraph.shape[0])
# 	shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
# 	return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor.float()]

# def smilesTordkit2dNorm(s):    
#     try:
#         generator = rdNormalizedDescriptors.RDKit2DNormalized()
#         features = np.array(generator.process(s)[1:])
#         NaNs = np.isnan(features)
#         features[NaNs] = 0
#     except:
#         print('descriptastorus not found this smiles: ' + s + ' convert to all 0 features')
#         features = np.zeros((200, ))
#     return np.array(features)
    
# def smilesTordkit2d(s):    
#     try:
#         generator = rdDescriptors.RDKit2D()
#         features = np.array(generator.process(s)[1:])
#         NaNs = np.isnan(features)
#         features[NaNs] = 0
#     except:
#         print('descriptastorus not found this smiles: ' + s + ' convert to all 0 features')
#         features = np.zeros((200, ))
#     return np.array(features)