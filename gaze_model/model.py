#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 10:43:36 2022

@author: shaohao
"""

import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F
import os

class DNN(nn.Module):
    def __init__(self, input_size, enc_size, dr, Batch_norm=False):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.enc_sizes = [self.input_size]
        self.enc_sizes.extend(enc_size)
        self.dropout = nn.Dropout(dr)

        linear_blocks = [nn.Linear(in_f, out_f, bias=True) for in_f, out_f
                         in zip(self.enc_sizes, self.enc_sizes[1:])]
        relu_blocks = [nn.ReLU() for i in range(len(linear_blocks)-1)]
        if Batch_norm:
            batch_norm_blocks = [nn.BatchNorm1d(out_f) for _, out_f in
                                 zip(self.enc_sizes, self.enc_sizes[1:-1])]
            network = []
            for idx, block in enumerate(linear_blocks):
                network.append(block)
                if idx < len(linear_blocks)-1:
                    network.append(block)
                    network.append(batch_norm_blocks[idx])
        else:
            network = []
            for idx, block in enumerate(linear_blocks):
                network.append(block)
                if idx < len(linear_blocks)-1:
                    network.append(self.dropout)
                    network.append(relu_blocks[idx])
                    
        # network.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*network)

    def forward(self, x):
        prediction = self.classifier(x)

        return prediction
