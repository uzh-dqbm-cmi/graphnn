import os
import numpy as np
import pandas as pd

import torch
# from torch.utils.data import Dataset, DataLoader
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset, Batch

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm as scpnorm
import pandas as pd
from .utilities import ModelScore, ReaderWriter, create_directory
from .chemfeatures import *

class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root=None,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='tdcDDI',
                 smiles_list=None,
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root
        
#         create_directory("raw", root)
#         create_directory("processed", root)

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])


#     def get(self, idx):
#         data = Data()
#         for key in self.data.keys:
#             item, slices = self.data[key], self.slices[key]
#             s = list(repeat(slice(None), item.dim()))
#             s[data.cat_dim(key, item)] = slice(slices[idx],
#                                                     slices[idx + 1])
#             data[key] = item[s]
#         return data


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')
        
#     def pairCollate(self, data_list):
#         r"""Collates a python list of data objects to the internal storage
#         format of :class:`torch_geometric.data.InMemoryDataset`."""
#         keys = data_list[0].keys
#         print("keys", keys)
#         data = data_list[0].__class__()

#         for key in keys:
#             data[key] = []
#         slices = {key: [0] for key in keys}

#         for item, key in product(data_list, keys):
#             data[key].append(item[key])
#             if isinstance(item[key], Tensor) and item[key].dim() > 0:
#                 cat_dim = item.__cat_dim__(key, item[key])
#                 cat_dim = 0 if cat_dim is None else cat_dim
#                 s = slices[key][-1] + item[key].size(cat_dim)
#             else:
#                 s = slices[key][-1] + 1
#             slices[key].append(s)

#         if hasattr(data_list[0], '__num_nodes__'):
#             data.__num_nodes__ = []
#             for item in data_list:
#                 data.__num_nodes__.append(item.num_nodes)

#         for key in keys:
#             item = data_list[0][key]
#             if isinstance(item, Tensor) and len(data_list) > 1:
#                 if item.dim() > 0:
#                     cat_dim = data.__cat_dim__(key, item)
#                     cat_dim = 0 if cat_dim is None else cat_dim
#                     data[key] = torch.cat(data[key], dim=cat_dim)
#                 else:
#                     data[key] = torch.stack(data[key])
#             elif isinstance(item, Tensor):  # Don't duplicate attributes...
#                 data[key] = data[key][0]
#             elif isinstance(item, int) or isinstance(item, float):
#                 data[key] = torch.tensor(data[key])

#             slices[key] = torch.tensor(slices[key], dtype=torch.long)

#         return data, slices

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == 'tdcDDI':
            X = ReaderWriter.read_data(os.path.join(self.raw_dir, 'X.pkl'))
            y = ReaderWriter.read_data(os.path.join(self.raw_dir, 'y.pkl'))
    
            for i,data in X.items():
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([y[i]])
                data_list.append(data)
                
        elif self.dataset == 'tdcSynergy':
            X = ReaderWriter.read_data(os.path.join(self.raw_dir, 'X.pkl'))
            y = ReaderWriter.read_data(os.path.join(self.raw_dir, 'y.pkl'))
            expression = ReaderWriter.read_data(os.path.join(self.raw_dir, 'expression.pkl'))
    
            for i,data in X.items():
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([y[i]])
                data.expression = torch.tensor([expression[i]])
                data_list.append(data)                 

        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
#         data_smiles_series = pd.Series(data_smiles_list)
#         data_smiles_series.to_csv(os.path.join(self.processed_dir,
#                                                'smiles.csv'), index=False,
#                                   header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
#         print("exit process(self)")

        
#         elif self.dataset == 'tox21':
#             smiles_list, rdkit_mol_objs, labels = \
#                 _load_tox21_dataset(self.raw_paths[0])
#             for i in range(len(smiles_list)):
#                 print(i)
#                 rdkit_mol = rdkit_mol_objs[i]
#                 ## convert aromatic bonds to double bonds
#                 #Chem.SanitizeMol(rdkit_mol,
#                                  #sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#                 data = mol_to_graph_data_obj_simple(rdkit_mol)
#                 # manually add mol id
#                 data.id = torch.tensor(
#                     [i])  # id here is the index of the mol in
#                 # the dataset
#                 data.y = torch.tensor(labels[i, :])
#                 data_list.append(data)
#                 data_smiles_list.append(smiles_list[i])

#         elif self.dataset == 'hiv':
#             smiles_list, rdkit_mol_objs, labels = \
#                 _load_hiv_dataset(self.raw_paths[0])
#             for i in range(len(smiles_list)):
#                 print(i)
#                 rdkit_mol = rdkit_mol_objs[i]
#                 # # convert aromatic bonds to double bonds
#                 # Chem.SanitizeMol(rdkit_mol,
#                 #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#                 data = mol_to_graph_data_obj_simple(rdkit_mol)
#                 # manually add mol id
#                 data.id = torch.tensor(
#                     [i])  # id here is the index of the mol in
#                 # the dataset
#                 data.y = torch.tensor([labels[i]])
#                 data_list.append(data)
#                 data_smiles_list.append(smiles_list[i])
            
#         elif self.dataset == 'clintox':
#             smiles_list, rdkit_mol_objs, labels = \
#                 _load_clintox_dataset(self.raw_paths[0])
#             for i in range(len(smiles_list)):
#                 print(i)
#                 rdkit_mol = rdkit_mol_objs[i]
#                 if rdkit_mol != None:
#                     # # convert aromatic bonds to double bonds
#                     # Chem.SanitizeMol(rdkit_mol,
#                     #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#                     data = mol_to_graph_data_obj_simple(rdkit_mol)
#                     # manually add mol id
#                     data.id = torch.tensor(
#                         [i])  # id here is the index of the mol in
#                     # the dataset
#                     data.y = torch.tensor(labels[i, :])
#                     data_list.append(data)
#                     data_smiles_list.append(smiles_list[i])


#         elif self.dataset == 'sider':
#             smiles_list, rdkit_mol_objs, labels = \
#                 _load_sider_dataset(self.raw_paths[0])
#             for i in range(len(smiles_list)):
#                 print(i)
#                 rdkit_mol = rdkit_mol_objs[i]
#                 # # convert aromatic bonds to double bonds
#                 # Chem.SanitizeMol(rdkit_mol,
#                 #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#                 data = mol_to_graph_data_obj_simple(rdkit_mol)
#                 # manually add mol id
#                 data.id = torch.tensor(
#                     [i])  # id here is the index of the mol in
#                 # the dataset
#                 data.y = torch.tensor(labels[i, :])
#                 data_list.append(data)
#                 data_smiles_list.append(smiles_list[i])
        
# def _load_tox21_dataset(input_path):
#     """
#     :param input_path:
#     :return: list of smiles, list of rdkit mol obj, np.array containing the
#     labels
#     """
#     input_df = pd.read_csv(input_path, sep=',')
#     smiles_list = input_df['smiles']
#     rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
#     tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
#        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
#     labels = input_df[tasks]
#     # convert 0 to -1
#     labels = labels.replace(0, -1)
#     # convert nan to 0
#     labels = labels.fillna(0)
#     assert len(smiles_list) == len(rdkit_mol_objs_list)
#     assert len(smiles_list) == len(labels)
#     return smiles_list, rdkit_mol_objs_list, labels.values

# def _load_hiv_dataset(input_path):
#     """
#     :param input_path:
#     :return: list of smiles, list of rdkit mol obj, np.array containing the
#     labels
#     """
#     input_df = pd.read_csv(input_path, sep=',')
#     smiles_list = input_df['smiles']
#     rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
#     labels = input_df['HIV_active']
#     # convert 0 to -1
#     labels = labels.replace(0, -1)
#     # there are no nans
#     assert len(smiles_list) == len(rdkit_mol_objs_list)
#     assert len(smiles_list) == len(labels)
#     return smiles_list, rdkit_mol_objs_list, labels.values

def _load_tdcDDI_dataset(smiles_list, labels):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
#     input_df = pd.read_csv(input_path, sep=',')
    input_df = pd.DataFrame({'smiles':smiles_list, 'labels':labels})
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['labels']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


# def _load_clintox_dataset(input_path):
#     """
#     :param input_path:
#     :return: list of smiles, list of rdkit mol obj, np.array containing the
#     labels
#     """
#     input_df = pd.read_csv(input_path, sep=',')
#     smiles_list = input_df['smiles']
#     rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

#     preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
#                                         rdkit_mol_objs_list]
#     preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
#                                 None for m in preprocessed_rdkit_mol_objs_list]
#     tasks = ['FDA_APPROVED', 'CT_TOX']
#     labels = input_df[tasks]
#     # convert 0 to -1
#     labels = labels.replace(0, -1)
#     # there are no nans
#     assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
#     assert len(smiles_list) == len(preprocessed_smiles_list)
#     assert len(smiles_list) == len(labels)
#     return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
#            labels.values
# # input_path = 'dataset/clintox/raw/clintox.csv'
# # smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)


# def _load_sider_dataset(input_path):
#     """
#     :param input_path:
#     :return: list of smiles, list of rdkit mol obj, np.array containing the
#     labels
#     """
#     input_df = pd.read_csv(input_path, sep=',')
#     smiles_list = input_df['smiles']
#     rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
#     tasks = ['Hepatobiliary disorders',
#        'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
#        'Investigations', 'Musculoskeletal and connective tissue disorders',
#        'Gastrointestinal disorders', 'Social circumstances',
#        'Immune system disorders', 'Reproductive system and breast disorders',
#        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
#        'General disorders and administration site conditions',
#        'Endocrine disorders', 'Surgical and medical procedures',
#        'Vascular disorders', 'Blood and lymphatic system disorders',
#        'Skin and subcutaneous tissue disorders',
#        'Congenital, familial and genetic disorders',
#        'Infections and infestations',
#        'Respiratory, thoracic and mediastinal disorders',
#        'Psychiatric disorders', 'Renal and urinary disorders',
#        'Pregnancy, puerperium and perinatal conditions',
#        'Ear and labyrinth disorders', 'Cardiac disorders',
#        'Nervous system disorders',
#        'Injury, poisoning and procedural complications']
#     labels = input_df[tasks]
#     # convert 0 to -1
#     labels = labels.replace(0, -1)
#     assert len(smiles_list) == len(rdkit_mol_objs_list)
#     assert len(smiles_list) == len(labels)
#     return smiles_list, rdkit_mol_objs_list, labels.value
        
    
    
    
# Source: https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html?highlight=pairdata#pairs-of-graphs
# class PairData(torch_geometric.data.Data):
#     def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
# #         print(type(self))
#         super(PairData, self).__init__()
#         self.edge_index_s = edge_index_s
#         self.x_s = x_s
#         self.edge_index_t = edge_index_t
#         self.x_t = x_t
        
#     def __inc__(self, key, value):
#         if key == 'edge_index_s':
#             return self.x_s.size(0)
#         if key == 'edge_index_t':
#             return self.x_t.size(0)
#         else:
#             return super().__inc__(key, value)     
        
class PairData(torch_geometric.data.Data):
    def __init__(self, data_a=None, data_b=None):
#         print(type(self))
        super(PairData, self).__init__()
    
        if ((data_a is not None) and (data_b is not None)): 
            self.edge_index_a = data_a.edge_index
            self.x_a = data_a.x
            self.edge_attr_a = data_a.edge_attr

            self.edge_index_b = data_b.edge_index
            self.x_b = data_b.x
            self.edge_attr_b = data_b.edge_attr
        
    def __inc__(self, key, value):
        if key == 'edge_index_a':
            return self.x_a.size(0)
        if key == 'edge_index_b':
            return self.x_b.size(0)
        else:
            return super().__inc__(key, value)
        
    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the graph."""
        if self.x_a is None:
            return 0
        return 1 if self.x_a.dim() == 1 else self.x_a.size(1)

    @property
    def num_features(self):
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self):
        r"""Returns the number of features per edge in the graph."""
        if self.edge_attr_a is None:
            return 0
        return 1 if self.edge_attr_a.dim() == 1 else self.edge_attr_a.size(1)
        
        
def pair_ids_to_pairdata(uniq_mol, pair, datafield):
#     print(a,b)
    data_a = uniq_mol.iloc[pair[0]][datafield]
    data_b = uniq_mol.iloc[pair[1]][datafield]
    return PairData(data_a, data_b)                

def get_X_all_pairdata(uniq_mol, pairs, datafield):
    return {key:pair_ids_to_pairdata(uniq_mol, pair, datafield) for key, pair in pairs.items()}        
        
def pair_ids_to_pairdata_synergy(uniq_mol, pair, datafield):
#     print(a,b)
    data_a = uniq_mol.loc[pair[0]][datafield]
    data_b = uniq_mol.loc[pair[1]][datafield]
    return PairData(data_a, data_b)                

def get_X_all_pairdata_synergy(uniq_mol, pairs, datafield):
    return {key:pair_ids_to_pairdata_synergy(uniq_mol, pair, datafield) for key, pair in pairs.items()}                        
        
        
        
        
        
        
        
        
        
class DeepAdrDataTensor(Dataset):

    def __init__(self, X_a, X_b, y):
        self.X_a = X_a # tensor.float32, (drug pairs, features)
        self.X_b = X_b # tensor.float32, (drug pairs, features)
        # drug interactions
        self.y = y  # tensor.float32, (drug pairs,)
        self.num_samples = self.y.size(0)  # int, number of drug pairs

    def __getitem__(self, indx):

        return(self.X_a[indx], self.X_b[indx], self.y[indx], indx)

    def __len__(self):
        return(self.num_samples)

class GIPDataTensor(Dataset):

    def __init__(self, X_a, X_b):
        self.X_a = X_a # tensor.float32, (drug pairs, gip features)
        self.X_b = X_b # tensor.float32, (drug pairs, gip features)
        # drug interactions
        self.num_samples = self.X_a.size(0)  # int, number of drug pairs

    def __getitem__(self, indx):

        return(self.X_a[indx], self.X_b[indx], indx)

    def __len__(self):
        return(self.num_samples)


class PartitionDataTensor(Dataset):

    def __init__(self, deepadr_datatensor, gip_datatensor, partition_ids, dsettype, fold_num):
        self.deepadr_datatensor = deepadr_datatensor  # instance of :class:`DeepAdrDataTensor`
        self.gip_datatensor = gip_datatensor # instance of :class:`GIPDataTensor`
        self.partition_ids = partition_ids  # list of indices for drug pairs
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.fold_num = fold_num  # int, fold number
        self.num_samples = len(self.partition_ids)  # int, number of docs in the partition

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        X_a, X_b, y, ddi_indx = self.deepadr_datatensor[target_id]
        X_a_gip, X_b_gip, gip_indx = self.gip_datatensor[target_id]
        # combine gip with other matrices
        X_a_comb = torch.cat([X_a, X_a_gip], axis=0)
        X_b_comb = torch.cat([X_b, X_b_gip], axis=0)
        return X_a_comb, X_b_comb, y, ddi_indx
        
    def __len__(self):
        return(self.num_samples)

def construct_load_dataloaders(dataset_fold, dsettypes, config, wrk_dir):
    """construct dataloaders for the dataset for one fold

       Args:
            dataset_fold: dictionary,
                          example: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                                    'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                                    'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                                    'class_weights': tensor([0.6957, 1.7778])
                                   }
            dsettype: list, ['train', 'validation', 'test']
            config: dict, {'batch_size': int, 'num_workers': int}
            wrk_dir: string, folder path
    """

    # setup data loaders
    data_loaders = {}
    epoch_loss_avgbatch = {}
    flog_out = {}
    score_dict = {}
    class_weights = {}
    for dsettype in dsettypes:
        if(dsettype == 'train'):
            shuffle = True
            class_weights[dsettype] = dataset_fold['class_weights']
        else:
            shuffle = False
            class_weights[dsettype] = None
        data_loaders[dsettype] = DataLoader(dataset_fold[dsettype],
                                            batch_size=config['batch_size'],
                                            shuffle=shuffle,
                                            num_workers=config['num_workers'])

        epoch_loss_avgbatch[dsettype] = []
        score_dict[dsettype] = ModelScore(0, 0.0, 0.0, 0.0, 0.0, 0.0)  # (best_epoch, auc, aupr, f1, precision, recall)
        if(wrk_dir):
            flog_out[dsettype] = os.path.join(wrk_dir, dsettype + ".log")
        else:
            flog_out[dsettype] = None

    return (data_loaders, epoch_loss_avgbatch, score_dict, class_weights, flog_out)

def ddi_dataframe_to_unique_drugs(df):
    # columns: ['Drug1_ID', 'Drug1', 'Drug2_ID', 'Drug2', 'Y']
    drug1 = pd.Series(df.Drug1.values,index=df.Drug1_ID).to_dict()
    drug2 = pd.Series(df.Drug2.values,index=df.Drug2_ID).to_dict()
    uniqe_drugs = {**drug1, **drug2}
    return pd.DataFrame(uniqe_drugs.items(), columns=['Drug_ID', 'Drug'])


def preprocess_features(feat_fpath, dsetname, fill_diag = None):
    if dsetname in {'DS1', 'DS3'}:
        X_fea = np.loadtxt(feat_fpath,dtype=float,delimiter=",")
    elif dsetname == 'DS2':
        X_fea = pd.read_csv(feat_fpath).values[:,1:]
    X_fea = X_fea.astype(np.float32)
    if fill_diag is not None:
        np.fill_diagonal(X_fea, fill_diag)
    return get_features_from_simmatrix(X_fea)

def get_features_from_simmatrix(sim_mat):
    """
    Args:
        sim_mat: np.array, mxm (drug pair similarity matrix)
    """
    r, c = np.triu_indices(len(sim_mat),1) # take indices off the diagnoal by 1
    return np.concatenate((sim_mat[r], sim_mat[c], sim_mat[r,c].reshape(-1,1), sim_mat[c,r].reshape(-1,1)), axis=1)

def preprocess_labels(interaction_fpath, dsetname):
    interaction_mat = get_interaction_mat(interaction_fpath, dsetname)
    return get_y_from_interactionmat(interaction_mat)

def get_y_from_interactionmat(interaction_mat):
    r, c = np.triu_indices(len(interaction_mat),1) # take indices off the diagnoal by 1
    return interaction_mat[r,c]

def generate_labels_df(uniq_mol, data):
    y = pd.DataFrame(0, columns=uniq_mol.Drug_ID, index=uniq_mol.Drug_ID)

    for index, row in data.iterrows():
        if (row.Drug1_ID in y.index and row.Drug2_ID in y.index):
            y.loc[row.Drug1_ID][row.Drug2_ID] = row.Y
            y.loc[row.Drug2_ID][row.Drug1_ID] = row.Y
            
    return y

def compute_gip_profile(adj, bw=1.):
    """approach based on Olayan et al. https://doi.org/10.1093/bioinformatics/btx731 """
    
    ga = np.dot(adj,np.transpose(adj))
    ga = bw*ga/np.mean(np.diag(ga))
    di = np.diag(ga)
    x =  np.tile(di,(1,di.shape[0])).reshape(di.shape[0],di.shape[0])
    d =x+np.transpose(x)-2*ga
    return np.exp(-d)

def compute_kernel(mat, k_bandwidth, epsilon=1e-9):
    """computes gaussian kernel from 2D matrix
    
       Approach based on van Laarhoven et al. doi:10.1093/bioinformatics/btr500
    
    """
    r, c = mat.shape # 2D matrix
    # computes pairwise l2 distance
    dist_kernel = squareform(pdist(mat, metric='euclidean')**2)
    gamma = k_bandwidth/(np.clip((scpnorm(mat, axis=1, keepdims=True)**2) * 1/c, a_min=epsilon, a_max=None))
    return np.exp(-gamma*dist_kernel)

def construct_sampleid_ddipairs(interaction_mat):
    # take indices off the diagnoal by 1
    r, c = np.triu_indices(len(interaction_mat),1)
    sid_ddipairs = {sid:ddi_pair for sid, ddi_pair in enumerate(zip(r,c))}
    return sid_ddipairs

def get_num_drugs(interaction_fpath, dsetname):
    if dsetname in {'DS1', 'DS3'}:
        interaction_matrix = np.loadtxt(interaction_fpath,dtype=float,delimiter=",")
    elif dsetname == 'DS2':
        interaction_matrix = pd.read_csv(interaction_fpath).values[:,1:]
    return interaction_matrix.shape[0]

def get_interaction_mat(interaction_fpath, dsetname):
    if dsetname in {'DS1', 'DS3'}:
        interaction_matrix = np.loadtxt(interaction_fpath,dtype=float,delimiter=",")
    elif dsetname == 'DS2':
        interaction_matrix = pd.read_csv(interaction_fpath).values[:,1:]
    return interaction_matrix.astype(np.int32)

def get_similarity_matrix(feat_fpath, dsetname):
    if dsetname in {'DS1', 'DS3'}:
        X_fea = np.loadtxt(feat_fpath,dtype=float,delimiter=",")
    elif dsetname == 'DS2':
        X_fea = pd.read_csv(feat_fpath).values[:,1:]
    X_fea = X_fea.astype(np.float32)
    return X_fea

def create_setvector_features(X, num_sim_types):
    """reshape concatenated features from every similarity type matrix into set of vectors per deepadr example"""
    e = X[np.newaxis, :, :]
    f = np.transpose(e, axes=(0, 2, 1))
    splitter = num_sim_types 
    g = np.concatenate(np.split(f, splitter, axis=1), axis=0)
    h = np.transpose(g, axes=(2,0, 1))
    return h

def get_stratified_partitions(y, num_folds=5, valid_set_portion=0.1, random_state=42):
    """Generate 5-fold stratified sample of drug-pair ids based on the interaction label

    Args:
        y: deepadr labels
    """
    skf_trte = StratifiedKFold(n_splits=num_folds, random_state=random_state, shuffle=True)  # split train and test
    
    skf_trv = StratifiedShuffleSplit(n_splits=2, 
                                     test_size=valid_set_portion, 
                                     random_state=random_state)  # split train and test
    data_partitions = {}
    X = np.zeros(len(y))
    fold_num = 0
    for train_index, test_index in skf_trte.split(X,y):
        
        x_tr = np.zeros(len(train_index))
        y_tr = y[train_index]

        for tr_index, val_index in skf_trv.split(x_tr, y_tr):
            tr_ids = train_index[tr_index]
            val_ids = train_index[val_index]
            data_partitions[fold_num] = {'train': tr_ids,
                                         'validation': val_ids,
                                         'test': test_index}
            
        print("fold_num:", fold_num)
        print('train data')
        report_label_distrib(y[tr_ids])
        print('validation data')
        report_label_distrib(y[val_ids])
        print('test data')
        report_label_distrib(y[test_index])
        print()
        fold_num += 1
        print("-"*25)
    return(data_partitions)

def validate_partitions(data_partitions, drugpairs_ids, valid_set_portion=0.1, test_set_portion=0.2):
    if(not isinstance(drugpairs_ids, set)):
        drugpairs_ids = set(drugpairs_ids)
    num_pairs = len(drugpairs_ids)
    test_set_accum = set([])
    for fold_num in data_partitions:
        print('fold_num', fold_num)
        tr_ids = data_partitions[fold_num]['train']
        val_ids = data_partitions[fold_num]['validation']
        te_ids = data_partitions[fold_num]['test']

        tr_val = set(tr_ids).intersection(val_ids)
        tr_te = set(tr_ids).intersection(te_ids)
        te_val = set(te_ids).intersection(val_ids)
        
        tr_size = len(tr_ids) + len(val_ids)
        # assert there is no overlap among train and test partition within a fold
        print('expected validation set size:', valid_set_portion*tr_size, '; actual test set size:', len(val_ids))
        assert len(tr_te) == 0
        print('expected test set size:', test_set_portion*num_pairs, '; actual test set size:', len(te_ids))
        print()
        assert np.abs(valid_set_portion*tr_size - len(val_ids)) <= 2
        assert np.abs(test_set_portion*num_pairs - len(te_ids)) <= 2
        for s in (tr_val, tr_te, te_val):
            assert len(s) == 0
        s_union = set(tr_ids).union(val_ids).union(te_ids)
        assert len(s_union) == num_pairs
        test_set_accum = test_set_accum.union(te_ids)
    # verify that assembling test sets from each of the five folds would be equivalent to all drugpair ids
    assert len(test_set_accum) == num_pairs
    assert test_set_accum == drugpairs_ids
    print("passed intersection and overlap test (i.e. train, validation and test sets are not",
          "intersecting in each fold and the concatenation of test sets from each fold is",
          "equivalent to the whole dataset)")

def report_label_distrib(labels):
    classes, counts = np.unique(labels, return_counts=True)
    norm_counts = counts/counts.sum()
    for i, label in enumerate(classes):
        print("class:", label, "norm count:", norm_counts[i])


def generate_partition_datatensor(deepadr_datatensor, gip_dtensor_perfold, data_partitions):
    datatensor_partitions = {}
    for fold_num in data_partitions:
        datatensor_partitions[fold_num] = {}
        gip_datatensor = gip_dtensor_perfold[fold_num]
        for dsettype in data_partitions[fold_num]:
            target_ids = data_partitions[fold_num][dsettype]
            datatensor_partition = PartitionDataTensor(deepadr_datatensor, gip_datatensor, target_ids, dsettype, fold_num)
            datatensor_partitions[fold_num][dsettype] = datatensor_partition
    compute_class_weights_per_fold_(datatensor_partitions)

    return(datatensor_partitions)

def build_datatensor_partitions(data_partitions, deepadr_datatensor):
    datatensor_partitions = generate_partition_datatensor(deepadr_datatensor, data_partitions)
    compute_class_weights_per_fold_(datatensor_partitions)
    return datatensor_partitions

def compute_class_weights(labels_tensor):
    classes, counts = np.unique(labels_tensor, return_counts=True)
    # print("classes", classes)
    # print("counts", counts)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels_tensor.numpy())
    return class_weights


def compute_class_weights_per_fold_(datatensor_partitions):
    """computes inverse class weights and updates the passed dictionary

    Args:
        datatensor_partitions: dictionary, {fold_num, int: {datasettype, string:{datapartition, instance of
        :class:`PartitionDataTensor`}}}}

    Example:
        datatensor_partitions
            {0: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                 'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                 'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>
                }, ..
            }
        is updated after computation of class weights to
            {0: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                 'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                 'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                 'class_weights': tensor([0.6957, 1.7778]),
                 }, ..
            }
    """

    for fold_num in datatensor_partitions:  # looping over the numbered folds
        dpartition = datatensor_partitions[fold_num]['train']
        partition_ids = dpartition.partition_ids
        labels = dpartition.deepadr_datatensor.y[partition_ids]
        datatensor_partitions[fold_num]['class_weights'] = torch.from_numpy(compute_class_weights(labels)).float()

def read_pickles(data_dir, device):

    # Read stored data structures
    data_partitions = ReaderWriter.read_data(os.path.join(data_dir, 'data_partitions.pkl'))
    # instance of :class:`DeepAdrDataTensor`
    deepadr_datatensor = ReaderWriter.read_tensor(os.path.join(data_dir, 'deepadr_datatensor.torch'), device)

    return data_partitions, deepadr_datatensor