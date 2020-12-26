"""
PyTorch implementation of a Spatial Transformer module.

References:
https://debuggercafe.com/spatial-transformer-network-using-pytorch/
https://github.com/aicaffeinelife/Pytorch-STN
"""
import os 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import logging

LOGGER = logging.getLogger(__name__)

class SpatialTransformer(nn.Module):
    """
    PyTorch implementation of a Spatial Transformer.
    """
    def __init__(self, in_channels):
        super(SpatialTransformer, self).__init__()

        # Localization net - simple CNN with ReLU activation
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Transformation regressor
        self.transformation = nn.Sequential(
            nn.Linear(64*3*3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3 * 2) # Affine transformation == 6 DOFs
        )

        # Initialize weights and biases of the last layer with the identity
        # affine transformation
        self.transformation[4].weight.data.zero_()
        self.transformation[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, x): 
        """
        Forward pass of the Spatial Transformer
        """
        xs = self.localization(x)
        LOGGER.info(f"Pre view size:{xs.size()}")
        xs = xs.view(-1, xs.size(1)*xs.size(2)*xs.size(3))
        
        # Calculate the transformation params
        theta = self.transformation(xs)

        # Resize into a 2x3 affine transformation matrix
        theta = theta.view(-1, 2, 3)

        # Generate affine grid for input sampling
        affine_grid = F.affine_grid(theta=theta, size=x.size())

        # Apply the spatial transformation and resample
        x = F.grid_sample(input=x, grid=affine_grid)

        # Return the transformed and resampled patches
        return x
