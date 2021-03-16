import pytorch_lightning as pl

from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets import GPData, SineData, ImgDataset, test_ImgDataset
import matplotlib.pyplot as plt

class NPBaseDataModule(pl.LightningDataModule):
    def __init__(self, dataset, test_dataset, num_workers=4, batch_size=4):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.x_dim = dataset.x_dim
        self.y_dim = dataset.y_dim

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.prepare_data()
        self.setup()

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
        self.val_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    """
    def test_dataloader(self):
        img_size = 28 if (self.dataset == 'mnist') else 32 
        if self.dataset == 'mnist':
            transform = transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor()
                                ])
            test_data = datasets.MNIST(path_to_data='../data', train=False,
                                transform=transform)
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        elif self.dataset == 'celeb':
            raise NotImplementedError
      return test_loader
    """

    def show_batch(self):
        # Helper function to visualize a batch of data, but unnecessary for training
        raise NotImplementedError


class NPDataModule(NPBaseDataModule):
    def __init__(self, dataset_type, num_workers=4, batch_size=4, **kwargs):
        assert dataset_type in ['sine', 'gpdata', 'mnist', 'celeb']
        self.dataset_type = dataset_type
        dataset = self._get_dataset(dataset_type, batch_size, **kwargs)
        test_dataset = self._get_test_dataset(dataset_type, batch_size, **kwargs)

        super().__init__(dataset=dataset, test_dataset=test_dataset, num_workers=num_workers, batch_size=batch_size)

    def _get_dataset(self, dataset_type, batch_size, **kwargs):
        if dataset_type == 'sine':
            dataset = SineData(**kwargs)
        elif dataset_type == 'gpdata':
            dataset = GPData(**kwargs)
        elif dataset_type == 'mnist':
            dataset = ImgDataset('mnist', batch_size, **kwargs)
        elif dataset_type == 'celeb':
            dataset = ImgDataset('celeb', batch_size, **kwargs)
        return dataset

    def _get_test_dataset(self, dataset_type, batch_size, **kwargs):
        if dataset_type == 'sine':
            test_dataset = SineData(**kwargs)
        elif dataset_type == 'gpdata':
            test_dataset = GPData(**kwargs)
        elif dataset_type == 'mnist':
            test_dataset = test_ImgDataset('mnist', batch_size, **kwargs)
        elif dataset_type == 'celeb':
            test_dataset = test_ImgDataset('celeb', batch_size, **kwargs)
        return test_dataset

    def show_batch(self):
        if self.dataset_type == 'gpdata':
            batch = next(iter(self.train_dataloader()))
            plt.scatter(batch['target_points_only_x'].cpu().numpy(),
                        batch['target_points_only_y'].cpu().numpy(),
                        c='#1f77b4')
            plt.scatter(batch['context_points_x'].cpu().numpy(),
                        batch['context_points_y'].cpu().numpy(),
                        c='#ff7f0e')
            return plt
        else:
            raise NotImplementedError
