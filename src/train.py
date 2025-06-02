import torch
from torch.utils.data import DataLoader
from data.memory_selection import ExemplarsDataset, HerdingExemplarsSelector, UncertaintyExemplarsSelector

class AdaptationTrainer:
    def __init__(self, model, memory_selector='herding', max_num_exemplars=1000, max_num_exemplars_per_class=100):
        self.model = model
        self.memory_dataset = ExemplarsDataset(
            max_num_exemplars=max_num_exemplars,
            max_num_exemplars_per_class=max_num_exemplars_per_class
        )
        self.memory_selector = self._get_memory_selector(memory_selector)
        
    def _get_memory_selector(self, selector_type):
        if selector_type == 'herding':
            return HerdingExemplarsSelector(self.memory_dataset)
        elif selector_type == 'uncertainty':
            return UncertaintyExemplarsSelector(self.memory_dataset)
        else:
            raise ValueError(f"Unknown memory selector type: {selector_type}")
            
    def train(self, train_loader, val_loader, num_epochs, optimizer, criterion):
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
            # Validation phase
            self.model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    
            # Update memory bank after each epoch
            self._update_memory(train_loader)
            
    def _update_memory(self, train_loader):
        """Update memory bank with new exemplars"""
        # Select new exemplars
        new_exemplars, new_labels = self.memory_selector(
            model=self.model,
            trn_loader=train_loader,
            transform=None,
            clean_memory=True
        )
        
        # Add new exemplars to memory bank
        self.memory_dataset.add_exemplars(new_exemplars, new_labels)
        
    def get_memory_loader(self, batch_size=32):
        """Get DataLoader for memory bank"""
        return DataLoader(
            self.memory_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        ) 