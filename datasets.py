import glob
import os
import numpy as np
import torch
import scipy.spatial
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class GPData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    lengthscale_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-1, 1]. <- could also change this later
    """
    def __init__(self, lengthscale_range=(0.1, 2), noise_range=(0.05, 1), num_samples=1000, num_points=100):
        self.is_img=False
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        min_l, max_l = lengthscale_range
        min_noise, max_noise = noise_range

        # Generate data
        rng = np.random.default_rng()
        self.xs = []
        self.ys = []
        for i in range(num_samples):
            points = rng.uniform(low=-1, high=1, size=(num_points, 1))
            x = np.sort(points, axis=0)
            self.xs.append(x)
            # print(x)
            
            lengthscale = (max_l - min_l) * rng.random() + min_l
            noise = (max_noise - min_noise) * rng.random() + min_noise

            cov = self.rbf_kernel(x, x, lengthscale, noise)

            # y = rng.multivariate_normal(np.zeros(num_points), cov)
            y = rng.multivariate_normal(np.zeros(num_points), cov)

            self.ys.append(np.expand_dims(y, 1))

    # will hopefully be able to add other kernels
    def rbf_kernel(self, xa, xb, lengthscale, noise):
        """rbf kernel"""
        # L2 distance (Squared Euclidian)
        sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
        return (noise**2)*np.exp(sq_norm/(lengthscale**2))

    def __getitem__(self, index):
        # slightly changed because not sure if it makes sense to choose points as linspace for training
        return torch.tensor(self.xs[index]).float(), torch.tensor(self.ys[index]).float()

    def __len__(self):
        return self.num_samples

class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.is_img = False
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class ImgDataset(Dataset):
    def __init__(self, data_type, path_to_data='../data', size=32, crop=89):
        self.is_img = True
        self.x_dim = 2
        self.y_dim = 1 if (data_type == 'mnist') else 3 
        if data_type == 'mnist':
            self.transforms = transforms.Compose([
                                transforms.Resize(min(size, 28)),
                                transforms.ToTensor()
                            ])
            self.ds = mnist(path_to_data=path_to_data, transform=self.transforms) 
        elif data_type == 'celeba':
            self.transforms = transforms.Compose([
                                transforms.CenterCrop(crop),
                                transforms.Resize(min(size, 32)),
                                transforms.ToTensor()
                            ])
            self.ds = CelebADataset(path_to_data=path_to_data, transform=self.transforms)

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)



def mnist(batch_size=16, size=28, path_to_data='../data', transform=None):
    """MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    # all_transforms = transforms.Compose([
    #     transforms.Resize(size),
    #     transforms.ToTensor()
    # ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=transform)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=transform)

    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data


def celeba(batch_size=16, size=32, crop=89, path_to_data='../celeba_data',
           shuffle=True):
    """CelebA dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image.

    crop : int
        Size of center crop. This crop happens *before* the resizing.

    path_to_data : string
        Path to CelebA data files.
    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    celeba_data = CelebADataset(path_to_data,
                                transform=transform)
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader


class CelebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.

        subsample : int
            Only load every |subsample| number of images.

        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        if os.path.isdir(f'{path_to_data}/celeba32/img_align_celeba'):
            path_to_data = f'{path_to_data}/celeba32/img_align_celeba'
        self.img_paths = glob.glob(path_to_data + '/*.jpg')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
