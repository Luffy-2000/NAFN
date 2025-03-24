import copy
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from data.dataset_config import ClassInfo
import data.utility as util


class NetworkingDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all biflows in memory"""

    def __init__(self, data, class_indices=None, is_unsupervised=False):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.seeds = data['s']
        self.class_indices = class_indices
        self.is_unsupervised = is_unsupervised

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index]
        y = self.labels[index]
        return x, y
    
    def merge(self, other):
        """Merges with another NetworkingDataset object"""
        # Encode other.labels from 0 to N
        ci = ClassInfo()
        le = LabelEncoder()
        le.fit(other.labels)
        y_re = torch.tensor(le.transform(other.labels))
        ci.data['new_classes'] = dict(zip(
            [str(e) for e in le.classes_], [str(e) for e in le.transform(le.classes_)]
        ))

        # Shift other.labels by max_existing_class + 1
        max_existing_class = torch.max(self.labels).item()
        y_re = y_re + max_existing_class + 1

        self.labels = torch.cat((self.labels, y_re))
        self.images = np.concatenate((self.images, other.images), axis=0)
        self.seeds = np.concatenate((self.seeds, other.seeds), axis=0)
        
        ci.data['all_classes'] = [arr.tolist() for arr in self.labels.unique(return_counts=True)]
        print(ci.data)
    

def _dataset_from_labels(x, y, class_set, indices=None, augs='', return_xy=False, is_unsupervised=False):
    """
    Generates a dataset from input features and labels based on the specified class set.

    Parameters:
        - x (array-like): Input features.
        - y (array-like): Labels.
        - class_set (array-like): The set of classes to include in the dataset.
        - indices (array-like, optional): The indices of the data samples. Defaults to None.
        - augs (``str``, optional): Augmentations to apply to the dataset. Defaults to '' (no aug).
        - return_xy (``bool``, optional): If True, returns input features and labels. 
          Defaults to False.
        - is_unsupervised (``bool``, optional): Whether to use unsupervised mode.
          Defaults to False.

    Returns:
        - NetworkingDataset or tuple: If `return_xy` is False, returns a NetworkingDataset object 
          containing input features, labels, and indices. If `return_xy` is True, returns a tuple 
          containing input features and labels.
    """
    if class_set is None:
        class_set = torch.tensor([])
    
    class_mask = (y[:, None] == class_set[None, :]).any(dim=-1)

    if isinstance(x, list):
        x = [e for e, m in zip(x, class_mask) if m]
    else:
        x = x[class_mask]
    y = y[class_mask]
    
    if return_xy:
        return x, y

    indices = ([util._get_random_state(idx) for idx in indices[class_mask]]
               if indices is not None else [])

    data = dict(zip(['x', 'y', 's'], [x, y, indices]))
    return NetworkingDataset(data, is_unsupervised=is_unsupervised)


def _class_split(fs_split, classes_per_set=[], sets_number=3):
    """
    Splits the dataset into specified class sets for training, validation, and testing.

    Parameters:
        - fs_split (``dict``): A dictionary containing keys ``train_classes``, ``val_classes``, 
            and ``test_classes``, each corresponding to the classes in the training, validation, 
            and test sets respectively.
        - classes_per_set (``list`` of ``int``, optional): A list specifying the number of 
            classes per set. 
        If not provided, the function returns the original class splits. Defaults to [].
        - sets_number (``int``, optional): The number of sets (typically 3 for train, val, test). 
        Defaults to 3.

    Returns:
        ``dict``: A dictionary containing keys ``trn``, ``val``, and ``tst``, each corresponding 
            to the class sets for training, validation, and testing respectively. 
    """
    
    if classes_per_set == []:
        train_classes = torch.tensor(fs_split['train_classes'])
        val_classes = torch.tensor(fs_split['val_classes']) if sets_number == 3 else None
        test_classes = torch.tensor(fs_split['test_classes'])
        
        return {
            'trn' : train_classes, 
            'val' : val_classes, 
            'tst' : test_classes
        }
    else:    
        classes = np.concatenate([fs_split[f'{k}_classes'] for k in ['train', 'val', 'test']])

        class_sets = []
        class_index = 0

        assert sum(classes_per_set) <= len(classes), (
            f'Classes per set ({sum(classes_per_set)}) exceed '
            f'the total available ({len(classes)})'
        )
        
        for c in classes_per_set:
            class_set = []
            for _ in range(c):
                class_set.append(classes[class_index])
                class_index += 1
                
            class_sets.append(torch.tensor(class_set)) 

        return {
            'trn' : util._safe_get(class_sets, 0, None), 
            'val' : util._safe_get(class_sets, 1, None) if len(classes_per_set) == 3 else None, 
            'tst' : util._safe_get(class_sets, 2 if len(classes_per_set) == 3 else 1, None)
        }   


def _balanced_hold_out(x, y, seed, min_samples_per_split=15, enc=True):
    
    x_train, x_val, x_test, y_train, y_val, y_test = util._train_val_test_split(x, y, seed, enc)
        
    for x_target, y_target in [(x_val, y_val), (x_test, y_test)]:

        labels, counters = np.unique(y_target, return_counts=True)
        indices = np.where(counters < min_samples_per_split)[0]
        minl = labels[indices]
        minc = counters[indices]
        
        for l,c in zip(minl, minc):
            delta = min_samples_per_split - c
            
            # Select the samples to remove from the train set
            indices_to_move  = np.where(y_train == l)[0]
            assert indices_to_move.size - delta >= min_samples_per_split, (
                f'Not enough samples in y_train for class {l} -> {indices_to_move.size} - {delta}'
            ) 
            indices_to_move  = np.random.choice(indices_to_move, size=delta, replace=False)
            
            # Copy the samples from the train set to the target set
            y_target = np.concatenate((y_target, y_train[indices_to_move]), axis=0)
            x_target = np.concatenate((x_target, x_train[indices_to_move]), axis=0)
            
            # Remove samples from the train set
            y_train = np.delete(y_train, indices_to_move, axis=0)
            x_train = np.delete(x_train, indices_to_move, axis=0)
            
    data_train = dict(zip(['x', 'y', 's'], [x_train, y_train, []]))
    data_test = dict(zip(['x', 'y', 's'], [x_test, y_test, []]))
    data_val = dict(zip(['x', 'y', 's'], [x_val, y_val, []]))
    
    train_set = NetworkingDataset(data_train)
    test_set = NetworkingDataset(data_test)
    val_set = NetworkingDataset(data_val)
    
    return train_set, val_set, test_set
        
    
def _hold_out(x, y, seed, enc=True):
    
    x_train, x_val, x_test, y_train, y_val, y_test = util._train_val_test_split(x, y, seed, enc)

    data_train = dict(zip(['x', 'y', 's'], [x_train, y_train, []]))
    data_test = dict(zip(['x', 'y', 's'], [x_test, y_test, []]))
    data_val = dict(zip(['x', 'y', 's'], [x_val, y_val, []]))
    
    train_set = NetworkingDataset(data_train)
    test_set = NetworkingDataset(data_test)
    val_set = NetworkingDataset(data_val)
    
    return train_set, val_set, test_set


def split(
    dc, num_pkts=None, fields=None, classes_per_set=[], shuffle_classes=False, is_fscil=False, seed=0, is_unsupervised=False
):
    """
    Splits the dataset into pre-training (train, val and test) and fine-tuning set sets 
    according to specified classes.

    Parameters:
        - dc (``dict``): The dataset dictionary containing keys 'path', 'fs_split', 
          and optionally 'label_column'.
        - num_pkts (``int``, optional): Number of packets for PSQ input. Defaults to None.
        - fields (``list`` of ``str``, optional): A list of fields to consider for PSQ input. 
          Defaults to None.
        - classes_per_set (``list`` of ``int``, optional): A list specifying the number of classes 
          per set. It should contain 0 or 2 elements. Defaults to [].
        - seed (``int``, optional): The random seed for dataset splitting. Defaults to 0.
        - is_unsupervised (``bool``, optional): Whether to use unsupervised pre-training. Defaults to False.

    Returns:
        ``tuple``: A tuple containing:
        - pt_ways (``int``): The number of classes in the pre-training set.
        - ft_ways (``int``): The number of classes in the fine-tuning set.
        - train_set (``NetworkingDataset``): The pre-training dataset.
        - test_set (``NetworkingDataset``): The testing dataset (for pt).
        - val_set (``NetworkingDataset``): The validation dataset (for pt).
        - finetune_set (``NetworkingDataset``): The fine-tuning dataset.
    """
    assert num_pkts and len(fields) > 0, (
        'NetworkingDataset requires the definition of num_pkts and fields, or both.'
    )
    assert len(classes_per_set) in [0, 2], (
        'classes_per_set must be empty or contain 2 elements.'
    )
    
    full_path = dc['path']
    label_column = dc.get('label_column', 'LABEL')
    fs_split = util._shuffle_dict_classes(dc['fs_split']) if shuffle_classes else dc['fs_split']
    
    x, y, _ = util._get_x_y(full_path, num_pkts, fields, label_column, seed)
    
    # Split classes
    class_sets = _class_split(fs_split, classes_per_set)
    
    # Split data for pre-training (both supervised and unsupervised use training classes)
    x_pt, y_pt = _dataset_from_labels(x, y, class_sets['trn'], return_xy=True)

    train_set, val_set, test_set = _balanced_hold_out(x_pt, y_pt, seed, enc=True)
    
    # Get ways
    ways = [len(class_sets['trn']), len(class_sets['tst'])]


    if is_unsupervised:
        # Convert the dataset to unsupervised mode
        train_set.is_unsupervised = True
        val_set.is_unsupervised = True
        test_set.is_unsupervised = True
        return ways, train_set, test_set, val_set, None
    

    # Create finetune set if needed
    if is_fscil:
        test_set_t1 = _dataset_from_labels(x, y, class_sets['tst'])
        finetune_set = copy.deepcopy(test_set)
        finetune_set.merge(test_set_t1)
    else:
        finetune_set = None
    
   
    return ways, train_set, test_set, val_set, finetune_set



    # # Split data
    # train_set = _dataset_from_labels(x, y, class_sets['trn'], is_unsupervised=is_unsupervised)
    # val_set = _dataset_from_labels(x, y, class_sets['val'], is_unsupervised=is_unsupervised)
    # test_set = _dataset_from_labels(x, y, class_sets['tst'], is_unsupervised=is_unsupervised)
    
    # # Get ways
    # ways = [len(class_sets['trn']), len(class_sets['tst'])]
    
    # return ways, train_set, test_set, val_set, None
