#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:31:57 2022

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
from model import ATT_CHAR_OTHER, ATT_CHAR_OTHER_V2, ATT_NEW, ATT_NEW_3
import torch.nn as nn
from torch.nn.modules.distance import PairwiseDistance
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score, precision_score
from data_loader import next_speaker_Dataset_attention

from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import random
import pdb
import csv
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# pdb.set_trace()
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,  default="new_model",help='model select')
parser.add_argument('--learning_rate', type=float,  default=5e-4,help='learning rate')
parser.add_argument('--dropout', type=float,  default=0.3,help='learning rate')
parser.add_argument('--batch_size', type=int,  default=128, help='batch size')
parser.add_argument('--epoch', type=int, default=30, help='padding')
parser.add_argument('--weighted', type=str,  default='yes',help='model select')
parser.add_argument('--gaze', type=str,  default='gaze_model',help='model select')
parser.add_argument('--label_type', type=str,  default='four',help='model select')
parser.add_argument('--char', type=str,  default='main_overall',help='model select')

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

skf = KFold(n_splits=FOLD)
#%%
if args.label_type == 'binary':
    CLASS_NUM = 1
elif args.label_type == 'four':
    CLASS_NUM = 4
LR = args.learning_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE = 'BEST_MODEL/{}/'.format(args.model)
RESULT_SAVE = 'result/{}.csv'.format(args.model)
IMG_SAVE = 'images/{}/'.format(args.model)

sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
TRAIN_RECORDING= sorted(set(sample['recording']))

sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
TEST_RECORDING = sorted(set(sample['recording']))

if not os.path.exists(MODEL_SAVE):
    os.mkdir(MODEL_SAVE)
    
if not os.path.exists(IMG_SAVE):
    os.mkdir(IMG_SAVE)

if not os.path.exists(RESULT_SAVE):
    with open(RESULT_SAVE, 'w', newline='') as x:
        writer = csv.writer(x)
        writer.writerow(['epoch', 'cv','label_mode', 'character_mode', 'model', 'LR', 'ACC', 'f1score', 'precision', 'UAR'])
#%% training main

ts_datasets = next_speaker_Dataset_attention(mode = 'test', gaze_fea_mode = args.gaze, recording_choose = TEST_RECORDING, 
                                             weighted = 'no', label_type = args.label_type, other_gaze_type = args.char)
ts_loaders = DataLoader(ts_datasets, batch_size=args.batch_size)    

for fold, (train_index, valid_index) in enumerate(skf.split(TRAIN_RECORDING)):
    
    if args.model == 'new_model_v1':
        model = ATT_CHAR_OTHER(talk_fea=1, talk_hidden=8, me_gaze_fea = 4, me_gaze_hidden = 16, gaze_fea = 1, gaze_hidden = 16, char_hidden=8,
                               layer_num = 1, tfm_head_talk = 2, tfm_head_gaze = 8, head_me_gaze = 2, out_dim = CLASS_NUM).to(device)
    elif args.model == 'new_model_v2':
        model = ATT_CHAR_OTHER_V2(talk_fea=1, talk_hidden=8, me_gaze_fea = 4, me_gaze_hidden = 16, gaze_fea = 1, gaze_hidden = 16, char_hidden=8,
                               layer_num = 1, tfm_head_talk = 2, tfm_head_gaze = 8, head_me_gaze = 2, out_dim = CLASS_NUM).to(device)
    elif args.model == 'new_model_v3':
        model = ATT_NEW(talk_fea=1, talk_hidden=4, me_gaze_fea = 4, me_gaze_hidden = 4, gaze_fea = 1, gaze_hidden = 4, char_hidden=4,
                               layer_num = 2, tfm_head_talk = 2, tfm_head_gaze = 2, head_me_gaze = 2, out_dim = CLASS_NUM).to(device)
    elif args.model == 'new_model':
        model = ATT_NEW_3(talk_fea=1, talk_hidden=4, me_gaze_fea = 4, me_gaze_hidden = 4, gaze_fea = 1, gaze_hidden = 4, char_hidden=4,
                               layer_num = 2, tfm_head_talk = 2, tfm_head_gaze = 2, head_me_gaze = 2, out_dim = CLASS_NUM).to(device)

    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    if args.label_type == 'binary':
        criterion = nn.BCELoss()
    elif args.label_type == 'four':
        criterion = nn.CrossEntropyLoss()
    
    model_save_path = os.path.join(MODEL_SAVE, 'cv_{}_{}_{}.sav'.format(fold, args.label_type, args.char))
    
    train_lst = [TRAIN_RECORDING[i] for i in train_index]
    valid_lst = [TRAIN_RECORDING[i] for i in valid_index]

    tr_datasets = next_speaker_Dataset_attention(mode = 'train', gaze_fea_mode = args.gaze, recording_choose = train_lst, 
                                                 weighted = args.weighted, label_type = args.label_type, other_gaze_type = args.char)
    if args.weighted == 'yes':
        weight = WeightedRandomSampler(tr_datasets.data_weight, len(tr_datasets))
        tr_loaders = DataLoader(tr_datasets, sampler=weight, batch_size=args.batch_size)
    else: 
        tr_loaders = DataLoader(tr_datasets, batch_size=args.batch_size, shuffle= True)
    
    # pdb.set_trace()
    valid_datasets = next_speaker_Dataset_attention(mode = 'train', gaze_fea_mode = args.gaze, recording_choose = valid_lst, 
                                                    weighted = 'no', label_type = args.label_type, other_gaze_type = args.char)
    valid_loaders = DataLoader(valid_datasets, batch_size=args.batch_size)
    
    train_loss_record = []
    valid_loss_record = []
    for epoch in range(args.epoch):
        
        model.train()
        train_loss = 0
        for step, (before, now, label, filename) in enumerate(tr_loaders): # talk_file_stretch, gaze_data, label, talk_file_path
            # pdb.set_trace()
            optimizer.zero_grad()
            
            talk_bef = before[0].unsqueeze(2).float().to(device)
            gaze_bef = before[1]
            char_bef = before[2]
            talk_now = now[0].unsqueeze(2).float().to(device)
            gaze_now = now[1]
            char_now = now[2]
            if args.label_type == 'binary':
                target_trans = label.type(torch.FloatTensor).to(device)
            elif args.label_type == 'four':
                target_trans = label.type(torch.LongTensor).to(device)
             
            _, output_trans = model.forward(talk_bef,
                                            gaze_bef[0].float().to(device),gaze_bef[1].float().to(device),
                                            gaze_bef[2].float().to(device),gaze_bef[3].float().to(device),char_bef.float().to(device),
                                            talk_now,
                                            gaze_now[0].float().to(device),gaze_now[1].float().to(device),
                                            gaze_now[2].float().to(device),gaze_now[3].float().to(device),char_now.float().to(device))
            
            if args.label_type == 'binary':
                output_trans = output_trans.reshape(-1)
            
            # pdb.set_trace()
            loss= criterion(output_trans, target_trans)
            loss.backward()
            optimizer.step()
            # pdb.set_trace()
            train_loss+=loss.data.cpu().numpy()
    
            if (step%100) == 0:
                print('epoch: {} step: {} loss: {:.4f}'.format(epoch, step, loss.data.cpu().numpy()))
        
        train_loss_record.append(train_loss/(step+1))
        
        valid_loss = 0
        with torch.no_grad():
            model.eval()
            best_val_loss = np.inf
            
            pred = []
            gt = []
            
            for step, (before, now, label, filename) in enumerate(valid_loaders):
                # pdb.set_trace()
                talk_bef = before[0].unsqueeze(2).float().to(device)
                gaze_bef = before[1]
                char_bef = before[2]
                talk_now = now[0].unsqueeze(2).float().to(device)
                gaze_now = now[1]
                char_now = now[2]
                if args.label_type == 'binary':
                    target_trans = label.type(torch.FloatTensor).to(device)
                elif args.label_type == 'four':
                    target_trans = label.type(torch.LongTensor).to(device)

                _, output_trans = model.forward(talk_bef,
                                                gaze_bef[0].float().to(device),gaze_bef[1].float().to(device),
                                                gaze_bef[2].float().to(device),gaze_bef[3].float().to(device),char_bef.float().to(device),
                                                talk_now,
                                                gaze_now[0].float().to(device),gaze_now[1].float().to(device),
                                                gaze_now[2].float().to(device),gaze_now[3].float().to(device),char_now.float().to(device))
                
                if args.label_type == 'binary':
                    output_trans = output_trans.reshape(-1)
                
                loss = criterion(output_trans, target_trans)
                valid_loss+=loss.data.cpu().numpy()
                # pdb.set_trace()
                if args.label_type == 'binary':
                    output_trans[output_trans>=0.5] = 1
                    output_trans[output_trans<0.5] = 0
                elif args.label_type == 'four':
                    _, output_trans = torch.max(output_trans.data,1)
                pred.extend(output_trans.cpu().tolist())
                gt.extend(label.cpu().tolist())

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
        
    plt.figure()
    plt.plot([i for i in range(len(train_loss_record))], train_loss_record, label = 'training loss')
    plt.plot([i for i in range(len(valid_loss_record))], valid_loss_record, label = 'valid loss')
    plt.title('{} fold {} loss'.format(args.model, fold+1))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(IMG_SAVE,'cv_{}_{}_{}.png'.format(fold, args.label_type, args.char)))
    # plt.show()
    
    model = torch.load(model_save_path)
    with torch.no_grad():
        model.eval()
        pred = []
        gt = []
        
        for step, (before, now, label, filename) in enumerate(ts_loaders):
            # pdb.set_trace()
            talk_bef = before[0].unsqueeze(2).float().to(device)
            gaze_bef = before[1]
            char_bef = before[2]
            talk_now = now[0].unsqueeze(2).float().to(device)
            gaze_now = now[1]
            char_now = now[2]
            if args.label_type == 'binary':
                target_trans = label.type(torch.FloatTensor).to(device)
            elif args.label_type == 'four':
                target_trans = label.type(torch.LongTensor).to(device)
             
            
            # pdb.set_trace()
            _, output_trans = model.forward(talk_bef,
                                            gaze_bef[0].float().to(device),gaze_bef[1].float().to(device),
                                            gaze_bef[2].float().to(device),gaze_bef[3].float().to(device),char_bef.float().to(device),
                                            talk_now,
                                            gaze_now[0].float().to(device),gaze_now[1].float().to(device),
                                            gaze_now[2].float().to(device),gaze_now[3].float().to(device),char_now.float().to(device))
            
            # pdb.set_trace()
            if args.label_type == 'binary':
                output_trans = output_trans.reshape(-1)
                output_trans[output_trans>=0.5] = 1
                output_trans[output_trans<0.5] = 0
            elif args.label_type == 'four':
                _, output_trans = torch.max(output_trans.data,1)
            pred.extend(output_trans.cpu().tolist())
            gt.extend(label.cpu().tolist())
            
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
        writer.writerow([args.epoch, fold, args.label_type, args.char, args.model, LR, ACC, f1score, precision, UAR])
  # writer.writerow(['epoch', 'label_mode', 'character_mode', 'model', 'LR', 'ACC', 'f1score', 'precision', 'UAR'])

