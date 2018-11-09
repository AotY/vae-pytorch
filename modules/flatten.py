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

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        """
        Args:
            input: [batch_size, c, h, w]

        return: [batch_size, c * h * w]

        """

        return input.view(input.size(0), -1)


