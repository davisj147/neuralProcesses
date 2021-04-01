import glob
import os
import numpy as np
import torch
import scipy.spatial
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from sklearn.gaussian_process.kernels import RBF, Matern, ExpSineSquared
from random import sample

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

    def __init__(self, num_context_points=20, num_target_points=80,
                 lengthscale_range=(0.25, 0.5), sigma_range=(1, 1), period_range=(1, 1),
                 num_samples=10000, num_points=100, kernel='rbf', shuffle_context_position=False):

        assert (kernel in ['rbf', 'periodic', 'matern'])
        self.is_img = False
        self.img_size = None  ##
        self.num_samples = num_samples
        self.num_context_points = num_context_points
        self.num_target_points = num_target_points
        self.num_points = num_points
        self.shuffle_context_position = shuffle_context_position
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        self.kernel = kernel

        self.min_l, self.max_l = lengthscale_range
        self.min_sigma, self.max_sigma = sigma_range
        self.min_p, self.max_p = period_range

        self._generate_data()
        # self._split_data_to_context_target_points()

    def _generate_data(self):
        # Generate data
        rng = np.random.default_rng()
        self.xs = []
        self.ys = []
        for i in range(self.num_samples):
            x, y, _, _, _ = self.generate_gp_sample(rng)

            self.xs.append(x)
            self.ys.append(np.expand_dims(y, 1))
    
    def generate_gp_sample(self, rng, bounds=(-1,1)):
        period=0
        x = np.linspace(bounds[0], bounds[1], self.num_points).reshape((-1,1))        
        lengthscale = (self.max_l - self.min_l) * rng.random() + self.min_l
        sigma = (self.max_sigma - self.min_sigma) * rng.random() + self.min_sigma

        if self.kernel == 'rbf':
            cov = self.rbf_kernel(x, x, lengthscale, sigma)
        if self.kernel == 'matern':
            cov = self.matern_kernel(x, x, lengthscale, sigma)
        if self.kernel == 'periodic':
            period = (self.max_p - self.min_p) * rng.random() + self.min_p
            cov = self.periodic_kernel(x, x, lengthscale, sigma, period)
        y = rng.multivariate_normal(np.zeros(self.num_points), cov)

        return x, y, lengthscale, sigma, period

    # will hopefully be able to add other kernels
    def rbf_kernel(self, xa, xb, lengthscale, sigma):
        """rbf kernel"""
        # L2 distance (Squared Euclidian)
        # sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
        # return (sigma**2)*np.exp(sq_norm/(lengthscale**2))
        kernel = RBF(length_scale=lengthscale)
        return sigma ** 2 * kernel(xa, xb)

    def periodic_kernel(self, xa, xb, lengthscale, sigma, period):
        # L1 distance
        # l1 = np.subtract.outer(xa, xb)
        # return (sigma**2)*np.exp((np.sin(np.pi * l1 / period)**2)/(lengthscale**2))
        kernel = ExpSineSquared(length_scale=lengthscale, periodicity=period)
        return sigma ** 2 * kernel(xa, xb)

    def matern_kernel(self, xa, xb, lengthscale, sigma, nu=1.5):
        kernel = Matern(length_scale=lengthscale, nu=nu)
        return sigma ** 2 * kernel(xa, xb)

    def __getitem__(self, index):
        # slightly changed because not sure if it makes sense to choose points as linspace for training
        return torch.tensor(self.xs[index]).float(), torch.tensor(self.ys[index]).float()
        # return self.data[index]

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
                 num_samples=1000, num_points=100, **kwargs):
        self.is_img = False
        self.img_size = None
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
            p = 2 * np.random.rand() + 1
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin((x - b)*p)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class ImgDataset(Dataset):
    def __init__(self, dataset_type, batch_size, path_to_data='../data', size=32, crop=89, **kwargs):
        self.batch_size = batch_size
        self.is_img = True
        self.img_size = 28 if (dataset_type == 'mnist') else 32
        self.x_dim = 2
        self.y_dim = 1 if (dataset_type == 'mnist') else 3
        if dataset_type == 'mnist':
            self.transforms = transforms.Compose([
                                transforms.Resize(self.img_size),
                                transforms.ToTensor()
                            ])
            self.ds = datasets.MNIST(path_to_data, train=True, transform=self.transforms) 
        elif dataset_type == 'celeb':
            self.transforms = transforms.Compose([
                                transforms.CenterCrop(150),
                                transforms.Resize(self.img_size),
                                transforms.ToTensor()
                            ])
            self.ds = CelebADataset(path_to_data,subsample=3, transform=self.transforms)

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)


class test_ImgDataset(Dataset):
    def __init__(self, dataset_type, batch_size, path_to_data='../data', **kwargs):
        self.batch_size = batch_size
        self.is_img = True
        self.img_size = 28 if (dataset_type == 'mnist') else 32
        self.x_dim = 2
        self.y_dim = 1 if (dataset_type == 'mnist') else 3
        if dataset_type == 'mnist':
            self.transforms = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ])
            self.ds = datasets.MNIST(path_to_data, train=False, transform=self.transforms)
        elif dataset_type == 'celeb':
            self.transforms = transforms.Compose([
                                transforms.CenterCrop(150),
                                transforms.Resize(self.img_size),
                                transforms.ToTensor()
                            ])
            self.ds = CelebADataset(path_to_data,subsample=3, transform=self.transforms)

    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)


# def mnist(batch_size=16, path_to_data='../data', transform=None):
#     """MNIST dataloader.
#     Parameters
#     ----------
#     batch_size : int
#     size : int
#         Size (height and width) of each image. Default is 28 for no resizing.
#     path_to_data : string
#         Path to MNIST data files.
#     """
#     # all_transforms = transforms.Compose([
#     #     transforms.Resize(size),
#     #     transforms.ToTensor()
#     # ])

#     train_data = datasets.MNIST(path_to_data, train=True, download=True,
#                                 transform=transform)
#     # test_data = datasets.MNIST(path_to_data, train=False,
#                                #transform=transform)

#     # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#     return train_data


# def celeba(batch_size=16, path_to_data='../celeba_data', transform=None):
#     """CelebA dataloader.
#     Parameters
#     ----------
#     batch_size : int
#     size : int
#         Size (height and width) of each image.
#     crop : int
#         Size of center crop. This crop happens *before* the resizing.
#     path_to_data : string
#         Path to CelebA data files.
#     """
#     #transform = transforms.Compose([
#     #    transforms.CenterCrop(crop),
#     #    transforms.Resize(size),
#     #    transforms.ToTensor()
#     #])

#     celeba_data = CelebADataset(path_to_data,subsample=1,
#                                transform=transform)
#     #celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
#     #                           shuffle=shuffle)
#     return celeba_data


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
        if os.path.isdir(f'../celebA/img_align_celeba'):
            path_to_data = f'../celebA/img_align_celeba'
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
