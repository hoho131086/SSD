#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:39:18 2022

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
from mdoel_gate import GATE_talk, GATE_candi, GATE_other
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
parser.add_argument('--model', type=str,  default="GATE_other",help='model select')
parser.add_argument('--learning_rate', type=float,  default=1e-3,help='learning rate')
parser.add_argument('--dropout', type=float,  default=0.3,help='learning rate')
parser.add_argument('--batch_size', type=int,  default=128, help='batch size')
parser.add_argument('--hidden', type=int,  default=16, help='batch size')
parser.add_argument('--layer_num', type=int,  default=2, help='batch size')
parser.add_argument('--loss', type=str, default='BCE', help='padding')
parser.add_argument('--epoch', type=int, default=30, help='padding')
parser.add_argument('--tfm_head', type=int, default=8, help='padding')
parser.add_argument('--date', type=str,  default="1220",help='model select')
parser.add_argument('--att_mode', type=str,  default="peter",help='model select')
parser.add_argument('--weighted', type=str,  default='yes',help='model select')
parser.add_argument('--func', type=str,  default='nn',help='model select')
parser.add_argument('--gaze', type=str,  default='gaze_model',help='model select')


args = parser.parse_args()
# pdb.set_trace()
print(args)
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

skf = KFold(n_splits=FOLD)

#%%
CLASS_NUM = 1
FEATURE_DIM = 11
LR = args.learning_rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_SAVE = 'BEST_MODEL/{}/'.format(args.date)
RESULT_SAVE = 'result/{}.csv'.format(args.model)
IMG_SAVE = 'images/{}/'.format(args.date)

sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
TRAIN_RECORDING= sorted(set(sample['recording']))

sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
TEST_RECORDING = sorted(set(sample['recording']))

if not os.path.exists(MODEL_SAVE):
    os.mkdir(MODEL_SAVE)
    
# if not os.path.exists(RESULT_SAVE):
#     os.mkdir(RESULT_SAVE)

if not os.path.exists(IMG_SAVE):
    os.mkdir(IMG_SAVE)

if not os.path.exists(RESULT_SAVE):
    with open(RESULT_SAVE, 'w', newline='') as x:
        writer = csv.writer(x)
        writer.writerow(['date', 'functional', 'fold', 'epoch', 'hidden', 'layer_num', 'dropout', 'LR', 'ACC', 'f1score', 'precision', 'UAR'])

#%% training main

ts_datasets = next_speaker_Dataset_attention(mode = 'test', gaze_fea_mode = args.gaze, recording_choose = TEST_RECORDING, weighted = 'no')
ts_loaders = DataLoader(ts_datasets, batch_size=args.batch_size)    

for fold, (train_index, valid_index) in enumerate(skf.split(TRAIN_RECORDING)):
    #  talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
    if args.model == 'GATE_talk':
        model = GATE_talk(talk_fea=1, talk_hidden=16, me_gaze_fea = 4, me_gaze_hidden = 16, gaze_fea = 1, gaze_hidden = 16,
                          layer_num = 1, tfm_head_talk = 2, tfm_head_gaze = 2, head_me_gaze = 2, out_dim = CLASS_NUM).to(device)
    if args.model == 'GATE_candi':
        model = GATE_candi(talk_fea=1, talk_hidden=16, me_gaze_fea = 4, me_gaze_hidden = 16, gaze_fea = 1, gaze_hidden = 16,
                           layer_num = 1, tfm_head_talk = 2, tfm_head_gaze = 2, head_me_gaze = 2, out_dim = CLASS_NUM).to(device)
    if args.model == 'GATE_other':
        model = GATE_other(talk_fea=1, talk_hidden=16, me_gaze_fea = 4, me_gaze_hidden = 16, gaze_fea = 1, gaze_hidden = 16,
                           layer_num = 1, tfm_head_talk = 2, tfm_head_gaze = 2, head_me_gaze = 2, out_dim = CLASS_NUM).to(device)

    

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    
    model_save_path = os.path.join(MODEL_SAVE, 'cv_{}_{}.sav'.format(fold, args.model))
    
    train_lst = [TRAIN_RECORDING[i] for i in train_index]
    valid_lst = [TRAIN_RECORDING[i] for i in valid_index]

    tr_datasets = next_speaker_Dataset_attention(mode = 'train', gaze_fea_mode = args.gaze, recording_choose = train_lst, weighted = args.weighted)
    if args.weighted == 'yes':
        weight = WeightedRandomSampler(tr_datasets.data_weight, len(tr_datasets))
        tr_loaders = DataLoader(tr_datasets, sampler=weight, batch_size=args.batch_size)
    else: 
        tr_loaders = DataLoader(tr_datasets, batch_size=args.batch_size, shuffle= True)
    
    # pdb.set_trace()
    valid_datasets = next_speaker_Dataset_attention(mode = 'train', gaze_fea_mode = args.gaze, recording_choose = valid_lst, weighted = 'no')
    valid_loaders = DataLoader(valid_datasets, batch_size=args.batch_size)
    
    train_loss_record = []
    valid_loss_record = []
    for epoch in range(args.epoch):
        
        model.train()
        train_loss = 0
        for step, (talk_x, gaze_x, label, filename) in enumerate(tr_loaders): # talk_file_stretch, gaze_data, label, talk_file_path
            optimizer.zero_grad()

            talk_x = talk_x.unsqueeze(2).float().to(device)
            target = label.float().unsqueeze(1).to(device)            
            output = model.forward(talk_x,
                                   gaze_x[0].float().to(device), gaze_x[1].float().to(device),
                                   gaze_x[2].float().to(device), gaze_x[3].float().to(device))
                
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # pdb.set_trace()
            train_loss+=loss.data.cpu().numpy()
    
            if (step%100) == 0:
                print('epoch: {} step: {} loss: {:.4f}'.format(epoch, step, loss.data.cpu().numpy()))
        
        train_loss_record.append(train_loss/(step+1.0))
        
        valid_loss = 0
        with torch.no_grad():
            model.eval()
            best_val_loss = np.inf
            
            pred = []
            gt = []
            
            for step, (talk_x, gaze_x, label, filename) in enumerate(valid_loaders):
                talk_x = talk_x.unsqueeze(2).float().to(device)
                target = label.float().unsqueeze(1).to(device)            
                
                output = model.forward(talk_x,
                                       gaze_x[0].float().to(device), gaze_x[1].float().to(device),
                                       gaze_x[2].float().to(device), gaze_x[3].float().to(device))
                
                
                loss = criterion(output, target)
                valid_loss+=loss.data.cpu().numpy()
                
                output[output>=0.5] = 1
                output[output<0.5] = 0
                pred.extend(output.cpu().tolist())
                gt.extend(label.cpu().tolist())
                # pdb.set_trace()

            valid_loss = valid_loss/(step+1.0)
            valid_loss_record.append(valid_loss)
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                torch.save(model, model_save_path)
                
            ACC = accuracy_score(gt, pred)
            precision = precision_score(gt, pred, average = 'macro', zero_division=0)
            UAR = recall_score(gt, pred, average = 'macro', zero_division=0)
            f1score = f1_score(gt, pred, average = 'macro', zero_division=0)
            
            print('\n========= CV: {}/{}  EPOCH: {}  Valid    =========='.format(fold+1, FOLD, epoch))
            print('valid loss: {:.4f}'.format(valid_loss))
            print(' UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f}  Precision: {:.4f}'.format(UAR, ACC, f1score, precision))
            CM = confusion_matrix(gt, pred)
            print(CM)
            print('=========================================================\n')
        
    #plt.figure()
    #plt.plot([i for i in range(len(train_loss_record))], train_loss_record, label = 'training loss')
    #plt.plot([i for i in range(len(valid_loss_record))], valid_loss_record, label = 'valid loss')
    #plt.title('{} fold {} loss'.format(args.model, fold+1))
    #plt.xlabel('epoch')
    #plt.ylabel('loss')
    #plt.legend()
    #plt.savefig('images/{}/train_two_class_{}_cv_{}.png'.format(args.date, args.model, fold))
    #plt.show()
    
    
    model = torch.load(model_save_path)
    with torch.no_grad():
        model.eval()
        pred = []
        gt = []
        
        for step, (talk_x, gaze_x, label, filename) in enumerate(ts_loaders):
            talk_x = talk_x.unsqueeze(2).float().to(device)
            target = label.float().unsqueeze(1).to(device)            
            
            output = model.forward(talk_x,
                                   gaze_x[0].float().to(device), gaze_x[1].float().to(device),
                                   gaze_x[2].float().to(device), gaze_x[3].float().to(device))
            
            output[output>=0.5] = 1
            output[output<0.5] = 0
            pred.extend(output.cpu().tolist())
            gt.extend(label.cpu().tolist())
            # pdb.set_trace()
            
        ACC = accuracy_score(gt, pred)
        precision = precision_score(gt, pred, average = 'macro', zero_division=0)
        UAR = recall_score(gt, pred, average = 'macro', zero_division=0)
        f1score = f1_score(gt, pred, average = 'macro', zero_division=0)
        
        print('========= {}/{}    Test    =========='.format(fold+1, FOLD))
        print(' UAR: {:.4f} ACC: {:.4f} F1 score: {:.4f} Precision: {:.4f}'.format(UAR, ACC, f1score, precision))
        CM = confusion_matrix(gt, pred)
        print(CM)
        print('========================================\n')
    
    with open(RESULT_SAVE, 'a', newline='') as x:
        writer = csv.writer(x)
        writer.writerow([args.date,args.func,fold,args.epoch,args.hidden,args.layer_num,args.dropout,LR,ACC,f1score,precision,UAR,args.model+'g'])
        


