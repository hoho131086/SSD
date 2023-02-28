#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:26:54 2022

@author: shaohao
"""

import joblib
import torch
import argparse
from os.path import exists
import numpy as np
import pandas as pd
# import random
# from sklearn.model_selection import KFold
import tqdm
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, precision_score
# from loader_four_class import next_speaker_Dataset_attention
from data_loader import next_speaker_Dataset_attention

from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import random
import pdb
import csv
# pdb.set_trace()
#%%
# warnings.filterwarnings("ignore")
FOLD=5
RANDSEED = 2021
np.random.seed(RANDSEED)
torch.manual_seed(RANDSEED)
torch.cuda.manual_seed(RANDSEED)
torch.cuda.manual_seed_all(RANDSEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(RANDSEED)
random.seed(RANDSEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% inference
cv = 1
ts_datasets = next_speaker_Dataset_attention(mode ='test',gaze_fea_mode='gaze_model',recording_choose = [],weighted = 'no', label_type='four', other_gaze_type='main_overall')
ts_loaders = DataLoader(ts_datasets, batch_size=1)    
model_save_path = "BEST_MODEL/new_model_v2/cv_4_four_main_overall.sav"
model = torch.load(model_save_path)
# pdb.set_trace()
output_dic = {}

with torch.no_grad():
    model.eval()
    pred = []
    gt = []
    pred_prob = []
    
    for step, (before, now, label, filename) in enumerate(ts_loaders):
        # pdb.set_trace()
        talk_bef = before[0].unsqueeze(2).float().to(device)
        gaze_bef = before[1]
        char_bef = before[2]
        talk_now = now[0].unsqueeze(2).float().to(device)
        gaze_now = now[1]
        char_now = now[2]
        target_trans = label.type(torch.LongTensor).to(device)               
        # pdb.set_trace()
        _, output_trans = model.forward(talk_bef,
                                        gaze_bef[0].float().to(device),gaze_bef[1].float().to(device),
                                        gaze_bef[2].float().to(device),gaze_bef[3].float().to(device), char_bef.float().to(device),
                                        talk_now,
                                        gaze_now[0].float().to(device),gaze_now[1].float().to(device),
                                        gaze_now[2].float().to(device),gaze_now[3].float().to(device), char_now.float().to(device))
        
        # pdb.set_trace()
        pred_prob.extend(output_trans.cpu().tolist())
        _, output_trans = torch.max(output_trans.data,1)
        # output_trans[output_trans>=0.5] = 1
        # output_trans[output_trans<0.5] = 0
        pred.extend(output_trans.cpu().tolist())
        gt.extend(label.cpu().tolist())
        
        member_id = filename[0].split('/')[-2]+'_'+filename[0].split('/')[-1].split('.')[0][-1]
        row_idx = str(filename[0].split('/')[-1].split('_')[1]).zfill(3)
        key = member_id+'_'+row_idx
        
        # pdb.set_trace()
    pred = np.array(pred)
    pred_prob = np.array(pred_prob)
    
    ACC = accuracy_score(gt, pred)
    precision = precision_score(gt, pred, average = 'macro', zero_division=0)
    UAR = recall_score(gt, pred, average = 'macro', zero_division=0)
    f1score = f1_score(gt, pred, average = 'macro', zero_division=0)
    
    print(' UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format( UAR, ACC, f1score, precision))
    CM = confusion_matrix(gt, pred)
    print(CM)
# joblib.dump(output_dic, 'embed_data/combine_me_cv_4.pkl')
#%%
print('\nBinary')
def trans_binary(lst):
    if lst == 0 or lst == 3:
        return 0
    elif lst == 1 or lst == 2:
        return 1
    else:
        pdb.set_trace()
    
pred_bin = pred.copy()
gt_bin = gt.copy()

pred_bin = [trans_binary(i) for i in pred_bin]
gt_bin = [trans_binary(i) for i in gt_bin]

ACC = accuracy_score(gt, pred)
precision = precision_score(gt_bin, pred_bin, average = 'macro', zero_division=0)
UAR = recall_score(gt_bin, pred_bin, average = 'macro', zero_division=0)
f1score = f1_score(gt_bin, pred_bin, average = 'macro', zero_division=0)

print(' UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format( UAR, ACC, f1score, precision))
CM = confusion_matrix(gt_bin, pred_bin)
print(CM)

#%%
# for thresh in range(30,100,5):
#     temp = pred_prob.copy()
#     temp[temp>=thresh/100.0] = 1
#     temp[temp<thresh/100.0] = 0
#     ACC = accuracy_score(gt, temp)
#     precision = precision_score(gt, temp, average = 'macro', zero_division=0)
#     UAR = recall_score(gt, temp, average = 'macro', zero_division=0)
#     f1score = f1_score(gt, temp, average = 'macro', zero_division=0)
    
#     print('thesh: {:.2f} UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format(thresh/100.0, UAR, ACC, f1score, precision))
#     CM = confusion_matrix(gt, temp)
#     print(CM)

