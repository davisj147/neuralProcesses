import pytorch_lightning as pl

from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

from datasets import GPData, SineData, mnist


class NPBaseDataModule(pl.LightningDataModule):
    def __init__(self, dataset, num_workers=4, batch_size=4):
        self.dataset = dataset
        self.x_dim = dataset.x_dim
        self.y_dim = dataset.x_dim

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.prepare_data()
        self.setup()

    def __init__(self, train_loader, test_loader, img_size, num_workers=4, batch_size=4):
        self.img_size = img_size # instead of x and y dim, use [1,28,28] for mnist / [3,32,32] for celeba

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_loader = train_loader
        self.val_loader = None
        self.test_loader = test_loader

        self.prepare_data()


    @property
    def collate_fn(self):
        return default_collate

    def prepare_data(self):
        pass

    def setup(self):
        # TODO split into train and validation set. At the moment, validation data is
        #  the same as training data
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       shuffle=True)
        self.val_loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        # No test set
        raise NotImplementedError

    def show_batch(self):
        # Helper function to visualize a batch of data, but unnecessary for training
        raise NotImplementedError


class NPDataModule(NPBaseDataModule):
    def __init__(self, dataset_type, num_workers=4, batch_size=4, **kwargs):
        assert dataset_type in ['sine', 'gpdata','mnist']
        self.dataset_type = dataset_type
        if dataset_type == 'sine' or 'gpdata':
            dataset = self._get_dataset(dataset_type, **kwargs)
            super().__init__(dataset=dataset, num_workers=num_workers, batch_size=batch_size)
        else:
            train_loader, test_loader = self._get_dataloader(dataset_type, **kwargs)
            super().__init__(train_loader=train_loader, test_loader=test_loader, num_workers=num_workers, batch_size=batch_size)



    def _get_dataset(self, dataset_type, **kwargs):
        if dataset_type == 'sine':
            dataset = SineData(**kwargs)
        elif dataset_type == 'gpdata':
            dataset = GPData(**kwargs)
        return dataset
    
    def _get_dataloader(self, dataset_type, **kwargs):
        if dataset_type == 'mnist':
            train_loader, test_loader = mnist()
            img_size = [1,28,28]
        elif dataset_type == 'celeba'
            ## to be installed
        
        return train_loader, test_loader, img_size
