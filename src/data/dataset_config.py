import json
from os.path import join

<<<<<<< HEAD
_BASE_DATA_PATH = "../../data"
=======
_BASE_DATA_PATH = "../data"
>>>>>>> 13490ca (Fix: Unsupervised Learning)

class ClassInfo:
    _instance = None
    data = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassInfo, cls).__new__(cls)
        return cls._instance
    
    def save_data(self, path):
        with open(f'{path}/classes_info.json', 'w') as f:
            json.dump(self.data, f)
            
    
dataset_config = {
    'iot_nid': {  
        'path': join(_BASE_DATA_PATH, 'iot_nid',
                     'iot-nidd_100pkts_6f_clean.parquet'),
        'class_order': [2, 7, 0, 6, 4, 3, 9, 5, 1, 8],  
        'fs_split': {  
            'train_classes': [2, 7, 0, 6], # benign, synflooding, ackflooding, portscanning
            'val_classes': [4, 3, 9], # httpflooding, hostdiscovery, udpflooding
            'test_classes': [5, 1, 8] # osversiondetection, arpspoofing, telnetbruteforce
        },
    },
    'cic2018': {  
        'path': join(_BASE_DATA_PATH, 'cic2018',
                     'cic2018_dataset_df_no_obf_20pkts_6feats_median_sampled_no_infiltration_clean_330ts.parquet'),
        'class_order': [8, 9, 5, 1, 2, 0, 6, 10, 3, 7, 4, 11],  
        'fs_split': {  
            'train_classes': [8, 9, 5, 1, 2, 0], 
            'val_classes': [6, 10, 3],
            'test_classes': [7, 4, 11]
        },
        'label_column': 'LABEL_FULL', 
    }
}
