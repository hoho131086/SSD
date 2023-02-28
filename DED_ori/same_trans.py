#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:05:17 2022

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

from model import Trans_Encoder_clf as TF
from model_g import ATT_TALK, ATT_OTHER_GAZE, ATT_ME_GAZE, ATT_Combine
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, precision_score
from data_loader import next_speaker_Dataset_attention
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import random
import pdb
import csv
# pdb.set_trace()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,  default="ATT_TALK",help='model select')
parser.add_argument('--gaze', type=str,  default="gaze_model",help='model select')
parser.add_argument('--batch_size', type=int,  default=1, help='model select')
args = parser.parse_args()
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
cv = 4
ts_datasets = next_speaker_Dataset_attention(mode = 'test', gaze_fea_mode = args.gaze, recording_choose = [], weighted = 'no')
ts_loaders = DataLoader(ts_datasets, batch_size=args.batch_size)    
# model_save_path = "/homes/GPU2/shaohao/turn_taking/turn_changing/final/BEST_MODEL/cv_{}_ATT_Combine_Me_small_me_gaze.sav".format(cv)
model_save_path = "/homes/GPU2/shaohao/turn_taking/turn_changing/final/BEST_MODEL/ANALYSIS/cv_{}_ATT_Combine_Me.sav".format(cv)

# model = torch.load(model_save_path)
trans_dic = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/final/DED/transit.pkl")
same_dic = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/final/DED/same.pkl")

pred = []
gt = []
check = []

trans_data = {}
same_data = {}

for step, (talk_x, gaze_x, label, filename) in enumerate(ts_loaders):
    # if (talk_x>0).sum()!=0: continue
    
    temp_path = filename[0]
    dic_key = temp_path.split('/')[-2]+'_'+str(os.path.basename(temp_path).split('.')[0][-1])
    current_num = str(int(os.path.basename(temp_path).split('_')[1]))
    temp_dic = {}
    temp_dic['talk'] = talk_x
    temp_dic['me_gaze'] = gaze_x[0]
    temp_dic['other_gaze'] = torch.cat([gaze_x[1].squeeze(0), gaze_x[2].squeeze(0), gaze_x[3].squeeze(0)], dim=1)
    
    if int(current_num) in trans_dic[dic_key] and int(current_num) in same_dic[dic_key]:
        pdb.set_trace()
    
    if int(current_num) in trans_dic[dic_key]:
        trans_data[dic_key+'_'+current_num] = temp_dic.copy()
    elif int(current_num) in same_dic[dic_key]:
        same_data[dic_key+'_'+current_num+'_talk'] = temp_dic.copy()
    else:
        pdb.set_trace()
    
#%%
for ind, key in enumerate(trans_data.keys()):
    
    if ind == 0:
        trans_talk = trans_data[key]['talk']
        trans_me = trans_data[key]['me_gaze']
        trans_other = trans_data[key]['other_gaze']
    # pdb.set_trace()
    else:
        trans_talk = torch.cat([trans_talk, trans_data[key]['talk']], dim=0)
        trans_me = torch.cat([trans_me, trans_data[key]['me_gaze']], dim=0)
        trans_other = torch.cat([trans_other, trans_data[key]['other_gaze']], dim=0)


for ind, key in enumerate(same_data.keys()):
    
    if ind == 0:
        same_talk = same_data[key]['talk']
        same_me = same_data[key]['me_gaze']
        same_other = same_data[key]['other_gaze']
    # pdb.set_trace()
    else:
        same_talk = torch.cat([same_talk, same_data[key]['talk']], dim=0)
        same_me = torch.cat([same_me, same_data[key]['me_gaze']], dim=0)
        same_other = torch.cat([same_other, same_data[key]['other_gaze']], dim=0)



















