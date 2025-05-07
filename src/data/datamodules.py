import pytorch_lightning as pl
from torch.utils import data

class PLDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_set,
        val_set = None,
        test_set = None,
        batch_size = 64,
        num_workers = 8, 
        pin_memory = True,
     ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

    def train_dataloader(self):
        if self.train_set is None:
            return None
        return data.DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        if self.val_set is None:
            return None
        return data.DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        if self.test_set is None:
            return None
        return data.DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )