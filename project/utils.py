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

from metrics.py import *


def compute_loss(x_pred, x_output, task_type):
    """
    Computes the loss between predicted and ground-truth data for either semantic or depth prediction tasks.

    Args:
        x_pred: predicted data tensor, of shape (batch_size, channels, height, width).
        x_output: ground-truth data tensor, of shape (batch_size, channels, height, width).
        task_type: string specifying the type of task, either 'semantic' or 'depth'.

    Returns:
        loss: scalar tensor representing the computed loss between x_pred and x_output.
    """

    device = x_pred.device  # Get the device where the tensors are stored (e.g. 'cpu' or 'cuda')

    # Create a binary mask to ignore undefined pixel spaces in x_output
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # Compute semantic loss using cross-entropy between x_pred and x_output
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    elif task_type == 'depth':
        # Compute depth loss using L1 norm between x_pred and x_output
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss

def trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, config, total_epoch=200):
  """
    Trains a multi-task model on semantic segmentation and depth estimation using the given train and test data loaders
    and hyperparameters. The model, optimizer and scheduler are expected to be defined outside the function and passed in.

    Args:
      train_loader: The data loader containing training data.
      test_loader: The data loader containing test data.
      multi_task_model: The multi-task model to train.
      device: The device to use for training (e.g. 'cpu' or 'cuda').
      optimizer: The optimizer to use for training the model.
      scheduler: The learning rate scheduler to use.
      config: A dictionary containing hyperparameters such as temperature and weighting scheme.
      total_epoch (optional): The total number of epochs to train the model. Default is 200.

    Returns:
      train_loss_sem, val_loss_sem, train_loss_dep, val_loss_dep
    """

  # Get the number of batches in training and test data
  train_batch = len(train_loader)
  test_batch = len(test_loader)
    
  # Set the temperature for Dynamic Weight Average (DWA) and initialize cost and lambda_weight
  T = config['temp']
  avg_cost = np.zeros([total_epoch, 12], dtype=np.float32)
  lambda_weight = np.ones([2, total_epoch])

  train_loss_sem = []
  val_loss_sem = []
  train_loss_dep = []
  val_loss_dep = []
    
  for index in range(total_epoch):
      cost = np.zeros(12, dtype=np.float32)

      # check if apply Dynamic Weight Average
      if config['weight'] == 'dwa':
          if index == 0 or index == 1:
              lambda_weight[:, index] = 1.0
          else:
              w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
              w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
              lambda_weight[0, index] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
              lambda_weight[1, index] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

      # Loop over all batches in the training data
      multi_task_model.train(True)
      train_dataset = iter(train_loader)
      conf_mat = ConfMatrix(multi_task_model.class_nb)

      for k in range(train_batch):
          train_data, train_label, train_depth = train_dataset.__next__()
          train_data, train_label = train_data.to(device), train_label.long().to(device)
          train_depth = train_depth.to(device)

          # Forward pass and compute training loss
          train_pred, logsigma = multi_task_model(train_data)
          optimizer.zero_grad()
          train_loss = [compute_loss(train_pred[0], train_label, 'semantic'),
                        compute_loss(train_pred[1], train_depth, 'depth')]
      
          # Compute weighted loss using either DWA or equal weights
          if config['weight'] == 'equal' or config['weight'] == 'dwa':
              loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(2)])
          else:
              loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(2))

          # Backward pass and optimization step
          loss.backward()
          optimizer.step()

          # Accumulate label prediction for every pixel in training images
          conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())
          
          # This block of code updates the cost and average cost for the current training batch
          cost[0] = train_loss[0].item() # is updated with the loss value for semantic segmentation
          cost[3] = train_loss[1].item() # is updated with the loss value for depth prediction
          cost[4], cost[5] = depth_error(train_pred[1], train_depth) # are updated with the depth error between the predicted depth and the ground truth depth.
          avg_cost[index, :6] += cost[:6] / train_batch # is updated with the current batch cost divided by the number of batches processed so far

      # compute mIoU and acc
     
      avg_cost[index, 1:3] = np.array(conf_mat.get_metrics())
      train_loss_sem.append(avg_cost[index, 0]) # batch loss on semantic task
      train_loss_dep.append(avg_cost[index, 3]) # batch loss on depth task
      
      # evaluate test data
      multi_task_model.eval()  # set the model to evaluation mode
      conf_mat = ConfMatrix(multi_task_model.class_nb)  # initialize confusion matrix
      with torch.no_grad():  # don't track gradients during evaluation
        test_dataset = iter(test_loader)  # create an iterator for the test data
        for k in range(test_batch):  # iterate through the test data in batches
          test_data, test_label, test_depth = test_dataset.__next__()  # get the next batch of test data
          test_data, test_label = test_data.to(device), test_label.long().to(device)  # move data to GPU
          test_depth = test_depth.to(device)

          # make predictions on test data and compute loss
          test_pred, _ = multi_task_model(test_data)  # make predictions
          test_loss = [compute_loss(test_pred[0], test_label, 'semantic'),
                     compute_loss(test_pred[1], test_depth, 'depth')]  # compute loss

          # update confusion matrix with predicted labels and compute cost metrics
          conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())  
          cost[6] = test_loss[0].item()  # semantic segmentation loss
          cost[9] = test_loss[1].item()  # depth estimation loss
          cost[10], cost[11] = depth_error(test_pred[1], test_depth)  # depth error metrics
          avg_cost[index, 6:] += cost[6:] / test_batch  # update average cost metrics

        # compute mIoU and acc
        avg_cost[index, 7:9] = conf_mat.get_metrics()

        val_loss_sem.append(avg_cost[index, 6]) # batch loss on semantic task
        val_loss_dep.append(avg_cost[index, 9]) # batch loss on depth task

      # Saving Model's General Check Point for Resuming training and inference
      if (index % 5 == 0): 
        torch.save({
            'epoch': index,
            'model_state_dict': multi_task_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, '/content/drive/MyDrive/DL/models/checkpoint_2.pth')
      
      scheduler.step()
      print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11]))
      
  return train_loss_sem, val_loss_sem, train_loss_dep, val_loss_dep