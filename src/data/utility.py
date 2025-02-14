import os
import torch
import random
import numpy as np
import pandas as pd
import socket, struct
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from data.dataset_config import ClassInfo


def _get_x_y(full_path, num_pkts, fields, label_column, seed, is_ml=False):
    """
    This function preprocesses a dataset and returns the input features (x) and labels (y).

    Parameters:
        - full_path (``str``): The full path to the dataset file.
        - num_pkts (``int``): The number of packets to consider.
        - fields (``list``): A list of fields to include in the input features.
        - label_column (``str``): The name of the column containing the labels.
        - seed (``int``): The random seed for reproducibility.
        - is_ml (``bool``, optional): If True, returns flattened input features for baselines. 
          Defaults to False.

    Returns:
        - tuple: A tuple containing input features (x), labels (y), and dataframe indices if `is_ml` 
          is False. If `is_ml` is True, returns only flattened input features and labels.
    """
    
    path = os.path.join(*full_path.split('/')[:-1])
    dataset_filename = full_path.split('/')[-1]
    dataset_extension = dataset_filename.split('.')[-1]
    
    prep_df_path = os.path.join(
        path, dataset_filename.replace(
            '.%s' % dataset_extension,
            '_prep%d.%s' % (seed, dataset_extension)
        )
    )
    
    if not os.path.exists(prep_df_path):
        # First time reading the dataset
        from sklearn.preprocessing import MinMaxScaler

        print('Processing dataframe...')
        # Read parquet
        df = pd.read_parquet(full_path)

        if 'ENC_LABEL' not in df:
            # Label encoding
            le = LabelEncoder()
            le.fit(df[label_column])
            df['ENC_LABEL'] = le.transform(df[label_column])
            label_conv = dict(zip(le.classes_, le.transform(le.classes_)))
            with open(os.path.join(path, 'classes_map.txt'), 'w') as fp:
                fp.write(str(label_conv))

        # Fields scaling & padding
        all_fields = ['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL', 'LOAD']
        existing_fields = [field for field in all_fields if field in df.columns]
        has_pad_col = 'FEAT_PAD' in df or 'LOAD_PAD' in df

        for field in existing_fields:
            mms = MinMaxScaler((0, 1))
            if has_pad_col:
                pad_field = 'FEAT_PAD' if field != 'LOAD' else 'LOAD_PAD'
                pad_value = 0.5 if field == 'DIR' else -1
                df[field] = df[[field, pad_field]].apply(
                    lambda x: np.concatenate((x[field], [pad_value] * x[pad_field])), axis=1)
            mms.fit(np.concatenate(df[field].values, axis=0).reshape(-1, 1))
            df['SCALED_%s' % field] = df[field].apply(
                lambda x: mms.transform(x.reshape(-1, 1)).reshape(-1)
            )

        df = df[['SCALED_%s' % field for field in existing_fields] + ['ENC_LABEL']]
        df.to_parquet(prep_df_path)
    else:
        print('WARNING: using pre-processed dataframe.')
        df = pd.read_parquet(prep_df_path)  # , engine='fastparquet')

    #  Get x and y
    columns = ['SCALED_%s' % field for field in fields]
    x = np.array([np.expand_dims(
        np.stack([r[:num_pkts] for r in row]).swapaxes(0, 1), axis=0) 
                    for row in df[columns].to_numpy()], dtype=np.float32)

    y = [label for label in df['ENC_LABEL']]
    y = torch.tensor(y)

    if is_ml:
        return _flatten(x), y
    else:
        return x, y, df.index


def _flatten(x):
    if isinstance(x, list):
        # Flatten and concatenate multi-modal inputs
        return np.array([np.concatenate((np.ravel(a), np.ravel(b))) for a,b in x])
    else:
        # Flatten single-modal input
        return np.array([np.ravel(a) for a in x]) 
  
 
def _train_val_test_split(x, y, seed, enc=True):
    if enc:
        ci = ClassInfo()
        le = LabelEncoder()
        le.fit(y)
        y_re = torch.tensor(le.transform(y))
        ci.data['old_classes'] = dict(zip(
            [str(e) for e in le.classes_], [str(e) for e in le.transform(le.classes_)]
        ))
    else:
        y_re = y
        
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_re, train_size=.7, random_state=seed, stratify=y_re)
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=.9, random_state=seed, stratify=y_train) 
    
    return x_train, x_val, x_test, y_train, y_val, y_test
    

def _get_random_state(idx):
    """
    Compute the random_state from the server IP address 
    """
    server_ip = idx.split(',')[2]
    try:
        ret = struct.unpack("!L", socket.inet_aton(server_ip))[0]
    except OSError as e:
        ret = struct.unpack("!QQ", socket.inet_pton(socket.AF_INET6, server_ip))[0]
    return ret 
    
    
def _safe_get(l, idx, default):
    try:
        return l[idx]
    except IndexError:
        return default 
   
    
def _shuffle_dict_classes(fs_split):
    all_classes = fs_split['train_classes'] + fs_split['val_classes'] + fs_split['test_classes']

    random.shuffle(all_classes)

    fs_split['train_classes'] = all_classes[:len(fs_split['train_classes'])]
    fs_split['val_classes'] = all_classes[
        len(fs_split['train_classes']):len(fs_split['train_classes']) + len(fs_split['val_classes'])
    ]
    fs_split['test_classes'] = all_classes[
        len(fs_split['train_classes']) + len(fs_split['val_classes']):
    ]

    return fs_split