#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
VAE Decoder
"""

import torch
import torch.nn as nn

from modules.unflatten import UnFlatten

class Decoder(nn.Module):
    def __init__(self, input_channels, z_size):
        super(Decoder, self).__init__()

        self.cnn_transpose = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(z_size, 32, (5, 5), 2, 0), # [batch_size, 32, 5, 5]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4, 4), 2, 0), #[batch_size, 16, 12, 12]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (3, 3), 2, 0), #[batch_size, 8, 25, 25]
            nn.ReLU(),
            nn.ConvTranspose2d(8, input_channels, (4, 4), 1, 0), #[batch_size, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Args:
            z: [batch_size, z_size]
        return: [batch_size, 1, h, w]
        """

        cnn_output = self.cnn_transpose(z)
        print(cnn_output.shape)

        return cnn_output





