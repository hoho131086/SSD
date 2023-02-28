#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:33:29 2022

@author: shaohao
"""
from torch.utils.data import Dataset

class multi(Dataset):

    def __init__(self, df, features):
        
        self.input = df[features].values.astype(float)
        self.labels = df['label'].values.astype(float)
        # import pdb;pdb.set_trace()

    def __getitem__(self, idx):
        
        
        return [self.input[idx, :], self.labels[idx]]

    def __len__(self):
        return self.input.shape[0]