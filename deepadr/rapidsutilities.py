import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def create_directory(folder_name, directory="current"):
    """create directory/folder (if it does not exist) and returns the path of the directory
       Args:
           folder_name: string representing the name of the folder to be created
       Keyword Arguments:
           directory: string representing the directory where to create the folder
                      if `current` then the folder will be created in the current directory
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)  # __file__ refers to utilities.py
    elif directory == "parent":
        path_current_dir = dirname(dirname(abspath(__file__)))
    else:
        path_current_dir = directory
    print("path_current_dir", path_current_dir)
        
    path_new_dir = os.path.normpath(os.path.join(path_current_dir, folder_name))
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)

class ReaderWriter(object):
    """class for dumping, reading and logging data"""
    def __init__(self):
        pass

    @staticmethod
    def read_or_dump_data(file_name, data_gen_fun, data_gen_params):
        if (isfile(file_name)):
            return ReaderWriter.read_data(file_name)
        else:
            data = data_gen_fun(*data_gen_params)
            ReaderWriter.dump_data(data, file_name)
            return data

    
    @staticmethod
    def dump_data(data, file_name, mode="wb"):
        """dump data by pickling
           Args:
               data: data to be pickled
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            pickle.dump(data, f)

    @staticmethod
    def read_data(file_name, mode="rb"):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)

    @staticmethod
    def dump_tensor(data, file_name):
        """
        Dump a tensor using PyTorch's custom serialization. Enables re-loading the tensor on a specific gpu later.
        Args:
            data: Tensor
            file_name: file path where data will be dumped
        Returns:
        """
        torch.save(data, file_name)

    @staticmethod
    def read_tensor(file_name, device):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               device: the gpu to load the tensor on to
        """
        data = torch.load(file_name, map_location=device)
        return data

    @staticmethod
    def write_log(line, outfile, mode="a"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(outfile, mode) as f:
            f.write(line)

    @staticmethod
    def read_log(file_name, mode="r"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(file_name, mode) as f:
            for line in f:
                yield line
                
                
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

def report_label_distrib(labels):
    classes, counts = np.unique(labels, return_counts=True)
    norm_counts = counts/counts.sum()
    for i, label in enumerate(classes):
        print("class:", label, "norm count:", norm_counts[i])