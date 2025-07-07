import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
import torch
from typing import List, Tuple, Optional
import numpy as np
from data import networking_dataset as netdat
from data.dataset_config import dataset_config
from data.datamodules import PLDataModule
from data.memory_selection import HerdingExemplarsSelector, UncertaintyExemplarsSelector
import random

class MemoryTaskDataset(l2l.data.TaskDataset):
    """Extend TaskDataset to support FSCIL memory"""
    def __init__(
        self, 
        dataset,  # finetune_set = old + new test set
        memory_dataset: Optional[torch.utils.data.Dataset] = None,
        task_transforms: List = None, 
        num_tasks: int = -1,
        memory_selector: str = 'herding',
        shots: int = 5,
        queries: int = 5,
        ways: int = 5,  # Number of new classes
        old_class_ids: Optional[List[int]] = None,
        new_class_ids: Optional[List[int]] = None,
        noise_label: bool = False,
        noise_ratio: float = 0.0,
    ):
        super().__init__(dataset, task_transforms, num_tasks)
        self.memory_dataset = memory_dataset
        self.memory_selector = self._get_memory_selector(memory_selector)
        self.shots = shots
        self.queries = queries
        self.ways = ways  # Number of new classes
        self.old_class_ids = old_class_ids
        self.new_class_ids = new_class_ids
        self.noise_label = noise_label
        self.noise_ratio = noise_ratio
    
    def _get_memory_selector(self, selector_type: str):
        """Get memory selector based on type"""
        if selector_type == 'herding':
            return HerdingExemplarsSelector
        elif selector_type == 'uncertainty':
            return UncertaintyExemplarsSelector
        else:
            raise ValueError(f"Unknown memory selector type: {selector_type}")

    def initialize_memory(self, model, train_set):
        """Select memory samples from training set"""
        if self.memory_selector is None:
            return
        selector = self.memory_selector(train_set, max_num_exemplars=1000, max_num_exemplars_per_class=self.shots)
        x, y = selector(
            model=model,
            trn_loader=torch.utils.data.DataLoader(train_set, batch_size=32),
            transform=None,
            clean_memory=True,
        )
        # Wrap memory as TensorDataset
        self.memory_dataset = torch.utils.data.TensorDataset(x, y)

    def add_label_noise_to_tensor(self, y, noise_ratio, num_classes):
        """
        Add label noise within each class: for each class, randomly select a portion of samples and change their labels to a wrong class, but keep the number of samples per class unchanged.
        """
        y = y.clone()
        for cls in range(num_classes):
            cls_indices = (y == cls).nonzero(as_tuple=True)[0]
            n_cls = len(cls_indices)
            num_noisy = int(n_cls * noise_ratio)
            if num_noisy == 0:
                continue
            noisy_indices = np.random.choice(cls_indices.cpu(), num_noisy, replace=False)
            for idx in noisy_indices:
                candidates = [l for l in range(num_classes) if l != cls]
                noisy_label = np.random.choice(candidates)
                y[idx] = noisy_label
        return y

    def sample_task(self):
        '''Build FSCIL task'''
        # Memory support → All old classes support
        if self.memory_dataset is not None and len(self.memory_dataset) > 0:
            memory_x, memory_y = zip(*[self.memory_dataset[i] for i in range(len(self.memory_dataset))])
            memory_x = torch.stack(memory_x)
            memory_y = torch.tensor(memory_y)
        else:
            memory_x = torch.empty(0)
            memory_y = torch.empty(0, dtype=torch.long)

        # New classes support → Sample from finetune_set: new_class_ids + shots per class
        new_support_x, new_support_y = self._sample_new_support()
        # Add label noise to support set if enabled
        if self.noise_label and self.noise_ratio > 0:
            new_support_y = self.add_label_noise_to_tensor(new_support_y, self.noise_ratio, self.ways)

        # Query set → Sample from finetune_set:
        #    - Old classes query (old class ids)
        #    - New classes query (new class ids)
        old_query_x, old_query_y = self._sample_old_query()
        new_query_x, new_query_y = self._sample_new_query()

        # Merge support
        support_data = torch.cat([memory_x, new_support_x], dim=0)
        support_labels = torch.cat([memory_y, new_support_y], dim=0)

        # Merge query
        query_data = torch.cat([old_query_x, new_query_x], dim=0)
        query_labels = torch.cat([old_query_y, new_query_y], dim=0)

        # Return Task
        x = torch.cat([support_data, query_data], dim=0)
        y = torch.cat([support_labels, query_labels], dim=0)
        return x, y
        # return task

    def _sample_new_support(self):
        '''Sample new class support'''
        # Sample from dataset (finetune_set) by new_class_ids with shots per class
        x_list, y_list = [], []
        for cls in self.new_class_ids:
            cls_indices = [i for i, (_, label) in enumerate(self.dataset) if label.item() == cls]
            selected_idx = random.sample(cls_indices, self.shots)
            x_cls, y_cls = zip(*[self.dataset[i] for i in selected_idx])
            x_cls = torch.stack([torch.tensor(xi) for xi in x_cls])
            y_cls = torch.tensor(y_cls)
            x_list.append(x_cls)
            y_list.append(y_cls)
        return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)

    def _sample_old_query(self):
        '''Sample old class query'''
        x_list, y_list = [], []
        for cls in self.old_class_ids:
            cls_indices = [i for i, (_, label) in enumerate(self.dataset) if label.item() == cls]
            selected_idx = random.sample(cls_indices, self.queries)
            x_cls, y_cls = zip(*[self.dataset[i] for i in selected_idx])
            x_cls = torch.stack([torch.tensor(xi) for xi in x_cls])
            y_cls = torch.tensor(y_cls)
            x_list.append(x_cls)
            y_list.append(y_cls)
        return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)

    def _sample_new_query(self):
        '''Sample new class query'''
        x_list, y_list = [], []
        for cls in self.new_class_ids:
            cls_indices = [i for i, (_, label) in enumerate(self.dataset) if label.item() == cls]
            selected_idx = random.sample(cls_indices, self.queries)
            x_cls, y_cls = zip(*[self.dataset[i] for i in selected_idx])
            x_cls = torch.stack([torch.tensor(xi) for xi in x_cls])
            y_cls = torch.tensor(y_cls)
            x_list.append(x_cls)
            y_list.append(y_cls)
        return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)


class EpisodeLoader(torch.utils.data.IterableDataset):
    def __init__(self, task_dataset, num_episodes):
        self.task_dataset = task_dataset
        self.num_episodes = num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            x, y = self.task_dataset.sample_task()
            yield x, y


def get_loaders(
    dataset, num_pkts, fields, queries, shots, num_tasks, 
    classes_per_set, shuffle_classes, is_fscil, seed,
    memory_selector: str = 'herding'
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
        - memory_selector (str): Type of memory selector ('herding' or 'uncertainty')
    """
    dc = dataset_config[dataset]
    
    ways, train_set, test_set, val_set, finetune_set = netdat.split(
        dc, num_pkts, fields, classes_per_set, shuffle_classes, is_fscil, seed
    )

    pretrain_datamodule = PLDataModule(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
    )
    
    
    return ways, pretrain_datamodule, finetune_set

def _get_taskset(dataset, ways, queries, shots, num_tasks, memory_selector='herding'):
    # Task size is equal to train/test_ways (N) * train/test_queries (K_query) + train/test_shots (K_support)
    dataset_md = l2l.data.MetaDataset(dataset)
    transforms = [
        NWays(dataset_md, ways),  # Samples N random classes per task
        KShots(dataset_md, queries + shots), # Samples K samples per class from the above N classes
        LoadData(dataset_md), # Loads a sample from the dataset
        # RemapLabels(dataset_md), # Remaps labels starting from zero
        ConsecutiveLabels(dataset_md) # Re-orders samples s.t. they are sorted in consecutive order 
    ]
    
    # Use MemoryTaskDataset
    return MemoryTaskDataset(
        dataset=dataset_md, 
        task_transforms=transforms, 
        num_tasks=num_tasks,
        memory_selector=memory_selector
    )