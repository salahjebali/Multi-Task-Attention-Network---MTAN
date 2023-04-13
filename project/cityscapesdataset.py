import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.utils.data.sampler as sampler

import os
import fnmatch 
import numpy as np
import random 
import matplotlib.pyplot as plt

class CityScapesDataset(Dataset):
    """
    A PyTorch dataset for loading pre-processed CityScapes data.

    Args:
        root (str): The root directory of the dataset.
        train (bool): Whether to load the training set or validation set.

    Attributes:
        train (bool): Whether to load the training set or validation set.
        root (str): The root directory of the dataset.
        data_path (str): The path to the directory containing the image, label, and
        depth data.
        data_len (int): The number of data points in the dataset.

    Methods:
        __getitem__(index): Loads an image (label, depth) from a pre-processed numpy
        file, changes its shape to fit the PyTorch tensor format, and returns it as a
        PyTorch tensor.
        __len__(): Returns the number of data points in the dataset.
    """

    def __init__(self, root, train=True):
        """
        Initializes the CityScapes dataset.

        Args:
            root (str): The root directory of the dataset.
            train (bool): Whether to load the training set or validation set.
        """
        self.train = train
        self.root = os.path.expanduser(root)

        if train:
            self.data_path = os.path.join(root, 'train')
        else:
            self.data_path = os.path.join(root, 'val')

        self.data_len = len(fnmatch.filter(os.listdir(os.path.join(self.data_path, 'image')), '*.npy'))

    def __getitem__(self, index):
        """
        Loads an image (label, depth) from a pre-processed numpy file, changes its shape
        to fit the PyTorch tensor format, and returns it as a PyTorch tensor.

        Args:
            index (int): The index of the data point to load.

        Returns:
            tuple: A tuple containing the image, label, and depth as PyTorch tensors.
        """
        image = torch.from_numpy(np.moveaxis(np.load(os.path.join(self.data_path, 'image', f'{index}.npy')), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label_7/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(os.path.join(self.data_path, 'depth', f'{index}.npy')), -1, 0))

        return image.float(), semantic.float(), depth.float()

    def __len__(self):
        """
        Returns the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return self.data_len
