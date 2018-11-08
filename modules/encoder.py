#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
VAE encoder
"""

import torch
import torch.nn as nn

from modules.flatten import Flatten


class Encoder(nn.Module):
    def __init__(self,
                 input_channels,
                 h_size,
                 z_size):
        """
        encoder input image to hidden vector
        minst image: 1 * 28 * 28
        image -> fc_layer -> (mu, logvar) + epsilon -> decoder -> image^
        mu:  h_size
        logvar: h_size
        epsilon: h_size
        z:
        """

        super(Encoder, self).__init__()

        self.input_channels = input_channels
        self.h_size = h_size
        self.z_size = z_size

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 8, (3, 3), 2, 0), #image: [batch_size, 8, 13, 13]
            nn.ReLU(),
            nn.Conv2d(8, 16, (4, 4), 2, 0), #image: [batch_size, 16, 10, 10]
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), 2, 0), #image: [batch_size, 32, 4, 4]
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2, 0), #image: [batch_size, 64, 1, 1]
            nn.ReLU(),
            Flatten()
        )

        self.mu_linear = nn.Linear(h_size, z_size)
        self.sigma_linear = nn.Linear(h_size, z_size)

    def epsilon_sample(self, size):
        """
        mu: [batch_size, z_size]
        """
        epsilon = torch.randn(size)
        return epsilon

    def forward(self, input):
        """
        input: [batch_size, input_channels, image_h, image_w]
        """
        cnn_output = self.cnn(input)
        print(cnn_output.shape)
        mu, logvar = self.mu_linear(cnn_output), self.sigma_linear(cnn_output)

        # reparameterize
        std = logvar.mul(0.5).exp_()
        epsilon = self.epsilon_sample()

        z = mu + std * epsilon

        return z, mu, logvar


