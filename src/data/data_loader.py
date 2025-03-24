import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels

from data import networking_dataset as netdat
from data.dataset_config import dataset_config
from data.datamodules import PLDataModule
<<<<<<< HEAD
=======
# from data.networking_dataset import UnsupervisedNetworkingDataset
>>>>>>> 13490ca (Fix: Unsupervised Learning)


def get_loaders(
    dataset, num_pkts, fields, queries, shots, num_tasks, 
<<<<<<< HEAD
    classes_per_set, shuffle_classes, is_fscil, seed
):
    
    dc = dataset_config[dataset]
    
    ways, train_set, test_set, val_set, finetune_set = netdat.split(
        dc, num_pkts, fields, classes_per_set, shuffle_classes, is_fscil, seed
    )
    
=======
    classes_per_set, shuffle_classes, is_fscil, seed, is_unsupervised=False
):
    """
    Get data loaders for training.

    Parameters:
        - dataset (str): Dataset name
        - num_pkts (int): Number of packets
        - fields (list): List of fields to use
        - queries (int): Number of query samples
        - shots (int): Number of support samples
        - num_tasks (int): Number of tasks
        - classes_per_set (list): Number of classes per set
        - shuffle_classes (bool): Whether to shuffle classes
        - is_fscil (bool): Whether to use FSCIL
        - seed (int): Random seed
        - is_unsupervised (bool): Whether to use unsupervised pre-training
    """
    dc = dataset_config[dataset]
    
    ways, train_set, test_set, val_set, finetune_set = netdat.split(
        dc, num_pkts, fields, classes_per_set, shuffle_classes, is_fscil, seed, is_unsupervised
    )

    if is_unsupervised:
        # Unsupervised pre-training uses training and validation sets
        unsupervised_datamodule = PLDataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,  
            batch_size=256,  
            num_workers=4  
        )
        return ways, unsupervised_datamodule, None

>>>>>>> 13490ca (Fix: Unsupervised Learning)
    pretrain_datamodule = PLDataModule(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
    )
<<<<<<< HEAD
=======

>>>>>>> 13490ca (Fix: Unsupervised Learning)
    finetune_taskset = _get_taskset(
        dataset=finetune_set,
        ways=sum(ways) if is_fscil else ways[1],
        queries=queries,
        shots=shots,
        num_tasks=num_tasks
    )

    return ways, pretrain_datamodule, finetune_taskset


def _get_taskset(dataset, ways, queries, shots, num_tasks):
    # Task size is equal to train/test_ways (N) * train/test_queries (K_query) + train/test_shots (K_support)
    dataset_md = l2l.data.MetaDataset(dataset)
    transforms = [
        NWays(dataset_md, ways),  # Samples N random classes per task
        KShots(dataset_md, queries + shots), # Samples K samples per class from the above N classes
        LoadData(dataset_md), # Loads a sample from the dataset
        # RemapLabels(dataset_md), # Remaps labels starting from zero
        ConsecutiveLabels(dataset_md) # Re-orders samples s.t. they are sorted in consecutive order 
    ]
    # If num_tasks = -1 infinite number of tasks will be produced
    # Creates sets of tasks from the dataset 
    return l2l.data.TaskDataset(dataset=dataset_md, task_transforms=transforms, num_tasks=num_tasks)