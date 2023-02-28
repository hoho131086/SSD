#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:32:29 2022

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
from glob import glob

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
trans_dic = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/final/DED/idx/transit.pkl")
ts_datasets = next_speaker_Dataset_attention(mode = 'test', gaze_fea_mode = args.gaze, recording_choose = [], weighted = 'no')
ts_loaders = DataLoader(ts_datasets, batch_size=args.batch_size)    


# model_save_path = "/homes/GPU2/shaohao/turn_taking/turn_changing/final/BEST_MODEL/cv_{}_ATT_Combine_Me_small_me_gaze.sav".format(cv)
# model_save_path = "/homes/GPU2/shaohao/turn_taking/turn_changing/final/BEST_MODEL/ANALYSIS/cv_{}_ATT_Combine_Me.sav".format(cv)

model_save_lst = glob('BEST_MODEL/ablation/*')
# pdb.set_trace()
model_save_lst = ['BEST_MODEL/1220/cv_2_GATE_candi.sav']
for model_save_path in model_save_lst:
    print(model_save_path)
    model = torch.load(model_save_path)
    output_dic = {}
    pred = []
    gt = []
    gt_true = []
    gt_false=[]
    check_true = []
    check_false = []
    same = []
    trans = []
    same_gt = []
    trans_gt = []
    
    
    with torch.no_grad():
        model.eval()
        '''
        step 979, 1036 -> other gaze high
        step 0 -> other gaze low
        step 457, 3157 -> talk high (300)
        step 9, 10 -> talk low (0)
        
        '''
        for step, (talk_x, gaze_x, label, filename) in enumerate(ts_loaders):
            talk_x = talk_x.unsqueeze(2).float().to(device)
            target = label.float().unsqueeze(1).to(device) 
            output = model.forward(talk_x,
                                   gaze_x[0].float().to(device), gaze_x[1].float().to(device),
                                   gaze_x[2].float().to(device), gaze_x[3].float().to(device))
            # pdb.set_trace()
            # if filename[0] in output_dic.keys():
            #     pdb.set_trace()
            # else:
            #     output_dic[filename[0]] = output.cpu().tolist()[0][0]
            
            # if output[0][0]==1:
            #     a = gaze_x[0].cpu().numpy()
            #     a = np.mean(a, axis=1)
            #     check_true.append(a)
            # elif output[0][0]==0:
            #     a = gaze_x[0].cpu().numpy()
            #     a = np.mean(a, axis=1)
            #     check_false.append(a)
            
            # if target[0][0]==1:
            #     a = gaze_x[0].cpu().numpy()
            #     a = np.mean(a, axis=1)
            #     gt_true.append(a)
            # elif target[0][0]==0:
            #     a = gaze_x[0].cpu().numpy()
            #     a = np.mean(a, axis=1)
            #     gt_false.append(a)
            # pdb.set_trace()
            output[output>=0.5] = 1
            output[output<0.5] = 0
            member_id = filename[0].split('/')[-2]+'_'+filename[0].split('/')[-1].split('.')[0][-1]
            row_idx = int(filename[0].split('/')[-1].split('_')[1])
            if row_idx in trans_dic[member_id]:
                trans.extend(output.cpu().numpy().reshape(-1,).tolist())
                trans_gt.extend(label.cpu().tolist())
            else:
                same.extend(output.cpu().numpy().reshape(-1,).tolist())
                same_gt.extend(label.cpu().tolist())
            
            # pdb.set_trace()
            # if member_id not in output_dic.keys():
            #     output_dic[member_id] = []
            #     output_dic[member_id].append(label.cpu().tolist()[0])
            # else:
            #     output_dic[member_id].append(label.cpu().tolist()[0])
            output_dic[filename[0]] = output.cpu().tolist()[0]
            
            pred.extend(output.cpu().numpy().reshape(-1,).tolist())
            gt.extend(label.cpu().tolist())
            # pdb.set_trace()
        pred = np.array(pred)
    # joblib.dump(output_dic, 'DED/data/model_output/output_GATE_other_cv2.pkl')
    
    uar = recall_score(gt,pred, average = 'macro')
    precision = precision_score(gt,pred, average = 'macro')
    f1 = f1_score(gt,pred, average = 'macro')
    acc = accuracy_score(gt,pred) 
    print('overall uar: {:.4f},  acc: {:.4f},  f1: {:.4f},  precision: {:.4f}'.format(uar, acc, f1, precision))
    
    uar = recall_score(trans_gt,trans, average = 'macro')
    precision = precision_score(trans_gt,trans, average = 'macro')
    f1 = f1_score(trans_gt,trans, average = 'macro')
    acc = accuracy_score(trans_gt,trans) 
    print('trans uar: {:.4f},  acc: {:.4f},  f1: {:.4f},  precision: {:.4f}'.format(uar, acc, f1, precision))
    
    uar = recall_score(same_gt,same, average = 'macro')
    precision = precision_score(same_gt,same, average = 'macro')
    f1 = f1_score(same_gt,same, average = 'macro')
    acc = accuracy_score(same_gt,same) 
    print('same uar: {:.4f},  acc: {:.4f},  f1: {:.4f},  precision: {:.4f}'.format(uar, acc, f1, precision))
    
    
    
    # check_true = np.array(check_true)
    # check_true = check_true.reshape(-1,4)
    # check_true = pd.DataFrame(check_true)
    # check_true.to_csv('check_true.csv')
    
    # check_false = np.array(check_false)
    # check_false = check_false.reshape(-1,4)
    # check_false = pd.DataFrame(check_false)
    # check_false.to_csv('check_false.csv')
    
    # gt_true = np.array(gt_true)
    # gt_true = gt_true.reshape(-1,4)
    # gt_true = pd.DataFrame(gt_true)
    # gt_true.to_csv('gt_true.csv')
    
    # gt_false = np.array(gt_false)
    # gt_false = gt_false.reshape(-1,4)
    # gt_false = pd.DataFrame(gt_false)
    # gt_false.to_csv('gt_false.csv')
    
    # pdb.set_trace()

    #%%  
    # 
    # for thresh in range(0, 105, 5):
    #     temp = pred.copy()
    #     temp[temp >= (thresh/100.0)] = 1
    #     temp[temp < (thresh/100.0)] = 0
    #     ACC = accuracy_score(gt, temp)
    #     precision = precision_score(gt, temp, average = 'macro', zero_division=0)
    #     UAR = recall_score(gt, temp, average = 'macro', zero_division=0)
    #     f1score = f1_score(gt, temp, average = 'macro', zero_division=0)
        
    #     print('Thresh: {:.2f} UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format(thresh/100.0, UAR, ACC, f1score, precision))
    #     CM = confusion_matrix(gt, temp)
    #     # print(CM)
    #     temp_trans = trans.copy()
    #     temp_trans = np.array(temp_trans)
    #     temp_trans[temp_trans >= (thresh/100.0)] = 1
    #     temp_trans[temp_trans < (thresh/100.0)] = 0
        
    #     temp_same = same.copy()
    #     temp_same = np.array(temp_same)
    #     temp_same = np.array(temp_same)
    #     temp_same[temp_same >= (thresh/100.0)] = 1
    #     temp_same[temp_same < (thresh/100.0)] = 0
        
    #     # ACC = accuracy_score(gt, pred)
    #     # precision = precision_score(gt, pred, average = 'macro', zero_division=0)
    #     # UAR = recall_score(gt, pred, average = 'macro', zero_division=0)
    #     # f1score = f1_score(gt, pred, average = 'macro', zero_division=0)
        
    #     # print('Overall UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format(UAR, ACC, f1score, precision))
    #     # CM = confusion_matrix(gt, pred)
    #     # print(CM)
        
    #     ACC = accuracy_score(trans_gt, temp_trans)
    #     precision = precision_score(trans_gt, temp_trans, average = 'macro', zero_division=0)
    #     UAR = recall_score(trans_gt, temp_trans, average = 'macro', zero_division=0)
    #     f1score = f1_score(trans_gt, temp_trans, average = 'macro', zero_division=0)
        
    #     print('Trans UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format(UAR, ACC, f1score, precision))
    #     CM = confusion_matrix(trans_gt, temp_trans)
    #     # print(CM)
        
    #     ACC = accuracy_score(same_gt, temp_same)
    #     precision = precision_score(same_gt, temp_same, average = 'macro', zero_division=0)
    #     UAR = recall_score(same_gt, temp_same, average = 'macro', zero_division=0)
    #     f1score = f1_score(same_gt, temp_same, average = 'macro', zero_division=0)
        
    #     print('Same UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format(UAR, ACC, f1score, precision))
    #     CM = confusion_matrix(same_gt, temp_same)
    #     # print(CM)
    #     print('='*30)


#%%
    # ran = 0.2
    # trans_indice = [idx for idx, val in enumerate(trans) if (val < (0.5-ran)) or (val > (0.5+ran))]
    # same_indice = [idx for idx, val in enumerate(same) if (val < (0.5-ran)) or (val > (0.5+ran))]
    
    # temp_gt = trans_gt.copy()
    # gt_select = [temp_gt[i] for i in trans_indice]
    # gt_skip = [val for i, val in enumerate(temp_gt) if i not in trans_indice]
    
    # temp_trans = trans.copy()
    # trans_select = [temp_trans[i] for i in trans_indice]
    # trans_skip = [val for i, val in enumerate(temp_trans) if i not in trans_indice]
    # trans_select = np.array(trans_select)
    # trans_skip = np.array(trans_skip)
    
    # trans_select[trans_select >= 0.5] = 1
    # trans_select[trans_select < 0.5] = 0
    # trans_skip[trans_skip >= 0.5] = 1
    # trans_skip[trans_skip < 0.5] = 0
    
    # UAR_select = recall_score(gt_select, trans_select, average = 'macro', zero_division=0)
    # UAR_skip = recall_score(gt_skip, trans_skip, average = 'macro', zero_division=0)
    # print('range: {}, certain count: {:.2f}, uncertrain: {:.2f}'.format(ran, len(gt_select)/len(trans_gt), len(gt_skip)/len(trans_gt)))
    # print('select: {:.4f},  skip: {:.4f}'.format(UAR_select, UAR_skip))
    
    # temp_gt = same_gt.copy()
    # gt_select = [temp_gt[i] for i in same_indice]
    # gt_skip = [val for i, val in enumerate(temp_gt) if i not in same_indice]
    
    # temp_same = same.copy()
    # same_select = [temp_same[i] for i in same_indice]
    # same_skip = [val for i, val in enumerate(temp_same) if i not in same_indice]
    # same_select = np.array(same_select)
    # same_skip = np.array(same_skip)
    
    # same_select[same_select >= 0.5] = 1
    # same_select[same_select < 0.5] = 0
    # same_skip[same_skip >= 0.5] = 1
    # same_skip[same_skip < 0.5] = 0
    
    # UAR_select = recall_score(gt_select, same_select, average = 'macro', zero_division=0)
    # UAR_skip = recall_score(gt_skip, same_skip, average = 'macro', zero_division=0)
    # print('range: {}, certain count: {:.2f}, uncertrain: {:.2f}'.format(ran, len(gt_select)/len(same_gt), len(gt_skip)/len(same_gt)))
    # print('select: {:.4f},  skip: {:.4f}'.format(UAR_select, UAR_skip))
    
    


