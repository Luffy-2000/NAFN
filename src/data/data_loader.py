import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
import torch
from typing import List, Tuple, Optional

from data import networking_dataset as netdat
from data.dataset_config import dataset_config
from data.datamodules import PLDataModule
from data.memory_selection import HerdingExemplarsSelector, UncertaintyExemplarsSelector

class MemoryTaskDataset(l2l.data.TaskDataset):
    """Extend TaskDataset to support memory management"""
    
    def __init__(
        self, 
        dataset, 
        memory_dataset: Optional[torch.utils.data.Dataset] = None,
        task_transforms: List = None, 
        num_tasks: int = -1,
        memory_selector: str = 'herding'
    ):
        super().__init__(dataset, task_transforms, num_tasks)
        self.memory_dataset = memory_dataset
        self.memory_selector = self._get_memory_selector(memory_selector)
        
    def _get_memory_selector(self, selector_type: str):
        """Get memory selector based on type"""
        if selector_type == 'herding':
            return HerdingExemplarsSelector
        elif selector_type == 'uncertainty':
            return UncertaintyExemplarsSelector
        else:
            raise ValueError(f"Unknown memory selector type: {selector_type}")
            
    def update_memory(self, model, new_data, exemplars_per_class: int):
        """Update memory bank with new data"""
        if self.memory_selector is None:
            return
            
        selector = self.memory_selector(new_data)
        memory_data = selector(
            model=model,
            trn_loader=torch.utils.data.DataLoader(new_data, batch_size=32),
            transform=None,
            exemplars_per_class=exemplars_per_class
        )
        self.memory_dataset = memory_data

    def sample_task(self):
        """Sample task including memory bank samples"""
        task = super().sample_task()
        
        if self.memory_dataset is not None:
            print(f"Memory dataset: {self.memory_dataset}")
            memory_task = self._sample_memory_task()
            print(f"Memory task: {memory_task}")
            task = self._merge_tasks(task, memory_task)
            
        return task
        
    def _sample_memory_task(self):
        """Sample task from memory bank"""
        if self.memory_dataset is None:
            return None
            
        # Apply same task transformations
        memory_md = l2l.data.MetaDataset(self.memory_dataset)
        for transform in self.task_transforms:
            memory_md = transform(memory_md)
            
        return memory_md
        
    def _merge_tasks(self, task, memory_task):
        """Merge original task and memory task"""
        if memory_task is None:
            return task
        
        # Merge data
        merged_data = torch.cat([task.data, memory_task.data], dim=0)
        merged_labels = torch.cat([task.labels, memory_task.labels], dim=0)
        
        # Create new task
        merged_task = l2l.data.Task(
            data=merged_data,
            labels=merged_labels
        )
        return merged_task

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
    
    # Use MemoryTaskDataset instead of original TaskDataset
    finetune_taskset = _get_taskset(
        dataset=finetune_set,
        ways=sum(ways) if is_fscil else ways[1],
        queries=queries,
        shots=shots,
        num_tasks=num_tasks,
        memory_selector=memory_selector
    )
    
    return ways, pretrain_datamodule, finetune_taskset

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