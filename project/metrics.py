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

class ConfMatrix:
    """
    A class for calculating metrics like accuracy and Intersection over Union (IoU)
    based on predicted and target values.

    Args:
        num_classes (int): The number of classes in the classification problem.

    Attributes:
        num_classes (int): The number of classes in the classification problem.
        conf_matrix (Tensor): The confusion matrix.

    Methods:
        update(pred, target): Updates the confusion matrix based on the predicted and
        target values.
        get_metrics(): Calculates the accuracy and IoU from the confusion matrix and
        returns their mean value.
    """

    def __init__(self, num_classes):
        """
        Initializes the ConfMatrix object.

        Args:
            num_classes (int): The number of classes in the classification problem.
        """
        self.num_classes = num_classes
        self.conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, pred, target):
        """
        Updates the confusion matrix based on the predicted and target values.

        Args:
            pred (Tensor): The predicted values.
            target (Tensor): The target values.
        """
        with torch.no_grad():
            valid_pixels = (target >= 0) & (target < self.num_classes)
            inds = self.num_classes * target[valid_pixels].to(torch.int64) + pred[valid_pixels]
            self.conf_matrix += torch.bincount(inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def get_metrics(self):
        """
        Calculates the accuracy and IoU from the confusion matrix and returns their mean
        value.

        Returns:
            tuple: A tuple containing the mean IoU and accuracy.
        """
        M = self.conf_matrix.float()
        acc = torch.diag(M).sum() / M.sum()
        IoU = torch.diag(M) / (M.sum(1) + M.sum(0) - torch.diag(M))
        return torch.mean(IoU).item(), acc.item()

def depth_error(x_pred, x_output):
    """
    Computes the average absolute and relative depth error between the predicted and ground truth depth maps.
    
    Args:
        x_pred (torch.Tensor): The predicted depth map tensor of shape (batch_size, 1, height, width).
        x_output (torch.Tensor): The ground truth depth map tensor of shape (batch_size, 1, height, width).
    
    Returns:
        Tuple of two floats: The average absolute depth error and the average relative depth error.
    """
    device = x_pred.device

    # Create a binary mask to mask out undefined pixel space.
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)

    # Select only the valid pixels.
    x_pred_true = x_pred[binary_mask]
    x_output_true = x_output[binary_mask]

    # Compute absolute and relative errors.
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = abs_err / x_output_true

    # Compute average errors and return as tuple.
    num_valid_pixels = binary_mask.sum()
    return abs_err.mean().item(), rel_err.mean().item() if num_valid_pixels > 0 else 0.0, 0.0