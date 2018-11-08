#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Flatten
"""

import torch
import torch.nn as nn

class UnFlatten(nn.Module):
    def __init__(self):
        super(UnFlatten, self).__init__()

    def forward(self, input):
        """
        Args:
            input: [batch_size, z_size]

        return: [batch_size, final_channels, 1, 1]

        """

        return input.view(input.size(0), -1, 1, 1)



