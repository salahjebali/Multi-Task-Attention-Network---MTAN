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

from cityscapesdataset import CityScapesDataset
from metrics import *
from utils import compute_loss, trainer

def conv_layer(channel):

      """
      Defines a convolutional layer in a neural network architecture.

      Args:
        channel (tuple): A tuple of two integers specifying the number of input and output channels for the layer.
        
      Returns:
        nn.Sequential: A PyTorch sequential module representing the convolutional layer with batch normalization and ReLU activation.
      """
      conv_layers = nn.Sequential(
        nn.Conv2d(in_channels = channel[0], out_channels = channel[1], kernel_size = 3, padding = 1),
        nn.BatchNorm2d(num_features = channel[1]),
        nn.ReLU(inplace = True),
        )

      return conv_layers

def att_layer(in_channels, out_channels):
    """
    Returns a PyTorch nn.Sequential module that implements a channel attention mechanism.

    The channel attention mechanism helps the neural network to focus on more important channels
    of the input feature maps and suppress the less important ones. The returned module consists
    of two 2D convolutional layers, two batch normalization layers, a ReLU activation function, and
    a sigmoid activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels for both convolutional layers.

    Returns:
        nn.Sequential: A PyTorch nn.Sequential module that implements the channel attention mechanism.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.Sigmoid()
    )

def init_weights(module):
    """
    This function initializes the weights of a neural network module.

    Args:
        module (torch.nn.Module): The module whose weights are to be initialized.

    Returns:
        None.
    """
    # If the module is a convolutional or linear layer, initialize the weights using the Kaiming Normal method
    # with fan-in and ReLU as the nonlinearity function. Initialize the biases to 0.
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(module.bias, 0)
    
    # If the module is a batch normalization layer, initialize the weights to 1 and biases to 0.
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class SegNetMTAN(nn.Module): 

  """
    A PyTorch implementation of the SegNet-Basic with Attention Modules for multi-task learning.
    This class develop 2 tasks: 
      - Segmentation Task 
      - Depth Task
  
    Args:
        None
        
    Attributes:
        predict_classes (int): Number of classes for segmentation task
        encoder_block (nn.ModuleList): A list of encoder blocks
        decoder_block (nn.ModuleList): A list of decoder blocks
        conv_encoder_block (nn.ModuleList): A list of convolution encoder blocks
        conv_decoder_block (nn.ModuleList): A list of convolution decoder blocks
        encoder_att (nn.ModuleList): A list of encoder attention modules
        decoder_att (nn.ModuleList): A list of decoder attention modules
        encoder_block_att (nn.ModuleList): A list of encoder attention blocks
        decoder_block_att (nn.ModuleList): A list of decoder attention blocks
        pred_segmentation (nn.Sequential): A sequential module for segmentation prediction
        pred_depth (nn.Sequential): A sequential module for depth prediction
        down_sampling (nn.MaxPool2d): A max pooling layer for downsampling
        up_sampling (nn.MaxUnpool2d): A max unpooling layer for upsampling
      
    Methods:
        __init__: Initializes the SegNet network by setting network parameters and encoder-decoder layers.
        forward(x): Defines the forward pass of the SegNetMTAN model
    """

  def __init__(self):
    super(SegNetMTAN, self).__init__()

    # Initializate network parameters
    filter = [64, 128, 256, 512, 512] # number of channels of each layer
    self.predict_classes = 7 # Number of classes for segmentation task

    # Initializate encoder and decoder blocks
    self.encoder_block = nn.ModuleList([
        self.conv_layer([in_filters, out_filters])
          for in_filters, out_filters in zip([3] + filter[:-1], filter)
    ])
    self.decoder_block = nn.ModuleList([
        self.conv_layer([in_filters, out_filters])
          for in_filters, out_filters in zip(filter[::-1] + [filter[-1]], filter[::-1])
    ])


    # Initialize convolution encoder-decoder layers
    self.conv_encoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
    self.conv_decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

    # Iterate over the range 4
    for i in range(4):
      # If i equals 0, append a single convolution layer to the encoder block
      if i == 0:
        self.conv_encoder_block.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
      # Otherwise, append a sequential block of two convolution layers to the encoder block
      else:
        self.conv_encoder_block.append(
            nn.Sequential(
            self.conv_layer([filter[i + 1], filter[i + 1]]),
            self.conv_layer([filter[i + 1], filter[i + 1]])
            )
        )
    
      # If i equals 0, append a single convolution layer to the decoder block
      if i == 0:
        self.conv_decoder_block.append(self.conv_layer([filter[i], filter[i]]))
      # Otherwise, append a sequential block of two convolution layers to the decoder block
      else:
        self.conv_decoder_block.append(
          nn.Sequential(
          self.conv_layer([filter[i], filter[i]]),
          self.conv_layer([filter[i], filter[i]])
            )
        )


    # Initialize attention modules layers
    self.encoder_att = nn.ModuleList([nn.ModuleList([att_layer(filter[0], filter[0])]) for _ in range(3)])
    self.decoder_att = nn.ModuleList([nn.ModuleList([att_layer(2 * filter[0], filter[0])]) for _ in range(3)])
    self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
    self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

    for j in range(3):
      for i in range(5):
        if i == 0:
          self.encoder_att[j].append(att_layer(filter[0], filter[0]))
          self.decoder_att[j].append(att_layer(2 * filter[0], filter[0]))
          self.encoder_block_att.append(self.conv_layer([filter[0], filter[1]]))
          self.decoder_block_att.append(self.conv_layer([filter[0], filter[0]]))
        elif i < 4:
          self.encoder_att[j].append(att_layer(2 * filter[i], filter[i]))
          self.decoder_att[j].append(att_layer(filter[i] + filter[i-1], filter[i-1]))
          self.encoder_block_att.append(self.conv_layer([filter[i], filter[i+1]]))
          self.decoder_block_att.append(self.conv_layer([filter[i], filter[i-1]]))
        else:
          self.encoder_block_att.append(self.conv_layer([filter[i], filter[i]]))
          self.decoder_block_att.append(self.conv_layer([filter[i], filter[i]]))

    # Initializate task specific layers
    self.pred_segmentation = nn.Sequential(
          nn.Conv2d(in_channels = filter[0], out_channels = filter[0], kernel_size = 3, padding = 1),
          nn.Conv2d(in_channels = filter[0], out_channels = self.predict_classes, kernel_size = 1, padding = 0),
          )

    self.pred_depth = nn.Sequential(
          nn.Conv2d(in_channels = filter[0], out_channels = filter[0], kernel_size = 3, padding = 1),
          nn.Conv2d(in_channels = filter[0], out_channels = 1, kernel_size = 1, padding = 0),
          )
    
    # Initialize pooling and unpooling layers for downsampling and upsampling respectively
    self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

    # Initialize weights of convolutional, batch norm, and linear layers
    self.init(init_weights)

  # Forward method
  def forward(self, x):
        if not isinstance(x, torch.Tensor):
          x = torch.tensor(x)

        encoder_output = [None] * 5
        decoder_output = [None] * 5
        maxpool_output = [None] * 5
        upsample_output = [None] * 5
        indices = [None] * 5

        for i in range(5):
          encoder_output[i] = [None] * 2
          decoder_output[-i-1] = [None] * 2

        # attention list for tasks
        att_encoder = [[ [0]*3 for _ in range(5) ] for _ in range(2)]
        att_decoder = [[ [0]*3 for _ in range(5) ] for _ in range(2)]
      
        ### SegNet : the global shared network ###
        
        # Encoder
        for i in range(5):
          input_x = x if i == 0 else maxpool_output[i - 1]
          encoder_output[i][0] = self.encoder_block[i](input_x)
          encoder_output[i][1] = self.conv_encoder_block[i](encoder_output[i][0])
          maxpool_output[i], indices[i] = self.down_sampling(encoder_output[i][1])

        # Decoder
        for i in range(5):
          input_x = maxpool_output[-1] if i == 0 else decoder_output[i - 1][-1]
          upsample_output[i] = self.up_sampling(input_x, indices[-i - 1])
          decoder_output[i][0] = self.decoder_block[-i - 1](upsample_output[i])
          decoder_output[i][1] = self.conv_decoder_block[-i - 1](decoder_output[i][0])


        ### Task Attention Modules ###

        # Encoder attention module
        for i in range(2):
          for j in range(5):
            if j == 0:
              att_encoder[i][j][0] = self.encoder_att[i][j](encoder_output[j][0])
            else:
              att_encoder[i][j][0] = self.encoder_att[i][j](
                torch.cat((encoder_output[j][0], att_encoder[i][j - 1][2]), dim=1)
            )
          att_encoder[i][j][1] = att_encoder[i][j][0] * encoder_output[j][1]
          att_encoder[i][j][2] = self.encoder_block_att[j](att_encoder[i][j][1])
          att_encoder[i][j][2] = F.max_pool2d(att_encoder[i][j][2], kernel_size=2, stride=2)

        # Decoder attention module
        for i in range(2):
          for j in range(5):
            if j == 0:
              att_decoder[i][j][0] = F.interpolate(
              att_encoder[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True
              )
            else:
              att_decoder[i][j][0] = F.interpolate(
              att_decoder[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True
            )
          att_decoder[i][j][0] = self.decoder_block_att[-j - 1](att_decoder[i][j][0])
          att_decoder[i][j][1] = self.decoder_att[i][-j - 1](
            torch.cat((upsample_output[j], att_decoder[i][j][0]), dim=1)
          )
          att_decoder[i][j][2] = att_decoder[i][j][1] * decoder_output[j][-1]

        # define task prediction layers
        task1_pred = F.log_softmax(self.pred_segmentation(att_decoder[0][-1][-1]), dim=1)
        task2_pred = self.pred_depth(att_decoder[1][-1][-1])

        return [task1_pred, task2_pred]

"""Training Configuration
Setting the device to run on and the number of epochs.
Setting the **temperature** as *temp* for *Dynamic Weight Average* equal to 2.0 and **weight mode** as *weight* as *DWA* or *equals*.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100 
config = {
    'temp': 2.0,
    'weight': 'dwa'
}

""" Model Definition"""

SegNet_mtan = SegNetMTAN().to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters of the model: {}'.format(count_parameters(SegNet_mtan)))

""" Optimizer and Scheduler
Setting **adam** as *optimizer* and then setting the **learnig rate scheduler** to adjust the *learning rate* based on the number of epoches
"""

optimizer = optim.Adam(SegNet_mtan.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

"""Dataset 
To launch the experiment, make sure that the drive is connected to the account, otherwise the dataset won't be found
"""

path = '/content/drive/MyDrive/DL/data'
train_set = CityScapes(root=path, train=True)
test_set = CityScapes(root=path, train=False)

batch_size = 8 # can be changed
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size = batch_size,
    shuffle = True,
    num_workers=2,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size = batch_size,
    shuffle = False,
    num_workers=2,
    pin_memory=True
)

"""##10 Training Launcher"""

print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR')
train_loss_sem, val_loss_sem, train_loss_dep, val_loss_dep = trainer(
                  train_loader,
                   test_loader,
                   SegNet_mtan,
                   device,
                   optimizer,
                   scheduler,
                   config,
                   epochs)

"""##11 Testing Launcher"""

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss - Semantic Segmentation")
plt.plot(val_loss_sem,label="val")
plt.plot(train_loss_sem,label="train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss - Semantic Segmentation")
plt.plot(val_loss_dep,label="val")
plt.plot(train_loss_dep,label="train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

valid_path = '/content/drive/MyDrive/DL/data/val/'
n_files = 10
# np.moveaxis(np.load(os.path.join(self.data_path, 'image', f'{index}.npy')), -1, 0))
files = random.sample(os.listdir(valid_path + 'image'), n_files)

segmentations = []
depths = []

for file in files:
  img = torch.from_numpy(np.expand_dims(np.moveaxis(np.load(valid_path + 'image/' + file),-1, 0), axis = 0))
  img = img.to(device)
  prediction, _ = SegNet_mtan(img.float())

  seg_prediction = np.moveaxis(prediction[0][0].cpu().detach().numpy(), 0, -1).argmax(2)
  depth_prediction = np.moveaxis(prediction[1][0].cpu().detach().numpy(), 0, -1)

  segmentations.append(seg_prediction)
  depths.append(depth_prediction)

"""##12 Results: Segmentation Labels"""

for file, seg_pred in zip(files, segmentations):
  seg_label = np.load(valid_path + 'label_7/' + file)

  plt.figure(figsize = (20, 10))
  plt.subplot(1, 2, 1)
  plt.imshow(seg_label)

  plt.subplot(1, 2, 2)
  plt.imshow(seg_pred)

for file, seg_pred in zip(files, segmentations):
    img = np.load(valid_path + 'image/' + file)
    
    plt.figure(figsize = (10,10))
    plt.imshow(img)
    plt.imshow(seg_pred, alpha = 0.7)

"""##13 Results: Depth Maps"""

for file, depth_pred in zip(files, depths):
    depth_label = np.load(valid_path + 'depth/' + file)
    
    plt.figure(figsize = (20,10))
    plt.subplot(1, 2, 1)
    plt.imshow(depth_label)

    plt.subplot(1, 2, 2)
    plt.imshow(depth_pred)

for file, depth_pred in zip(files, depths):
    img = np.load(valid_path + 'image/' + file)
    
    plt.figure(figsize = (10,10))
    plt.imshow(img)
    plt.imshow(depth_pred, alpha = 0.7)