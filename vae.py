#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Variational Auto-Encoder

x, z, x^
"""

import torch
import torch.nn as nn

from modules.encoder import Encoder
from modules.decoder import Decoder

class VAE(nn.Module):
    def __init__(self,
                 input_channels,
                 h_size,
                 z_size):
        super(VAE, self).__init__()

        self.encoder = Encoder(
            input_channels,
            h_size,
            z_size
        )

        self.decoder = Decoder(
            input_channels,
            z_size
        )


    def forward(self, input):
        z, mu, logvar = self.encoder(input)

        output = self.decoder(z)

        return output, mu, logvar

    def decode(self, z):
        output = self.decoder(z)
        return output




