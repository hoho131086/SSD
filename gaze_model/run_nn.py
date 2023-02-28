#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:04:13 2022

@author: shaohao
"""

from loader import multi
from model import DNN
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import os
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import joblib
import pdb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def display(bi_acc, f1, recall, precision, uar):
    print("Binary accuracy on test set is {:.4f}".format(bi_acc))
    print("F1-score on test set is {:.4f}".format(f1))
    print("Recall on test set is {:.4f}".format(recall))
    print("Precision on test set is {:.4f}".format(precision))
    print("UAR on test set is {:.4f}".format(uar))

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', dest='run_id', type=int, default=1)
parser.add_argument('--epochs', dest='epochs', type=int, default=30)
parser.add_argument('--features', dest='features', type=str, default='gaze_angle_x,gaze_angle_y,head_angle_x,head_angle_y,head_angle_z,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_1_z')     
parser.add_argument('--LR', dest='LR', type=float, default=5e-2)
parser.add_argument('--layer_num', dest='layer_num', type=int, default=4)
parser.add_argument('--node_num', dest='node_num', type=int, default=32)
parser.add_argument('--dropout', dest='dropout', type=float, default=0)
parser.add_argument('--output_dim', dest='output_dim', type=int, default=4)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)
parser.add_argument('--signiture', dest='signiture', type=str, default='multi')     
parser.add_argument('--cuda', dest='cuda', type=bool, default=True)
parser.add_argument('--model_path', dest='model_path', type=str, default='best_model')
parser.add_argument('--output_path', dest='output_path', type=str, default='path')
parser.add_argument('--target_group', dest='target_group', type=str, default='all')
parser.add_argument('--std', dest='std', type=str, default='no')
args = parser.parse_args()

# import pdb; pdb.set_trace()
# %%
CV_FOLD = 5
LR = args.LR
batch_sz = args.batch_size

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

skf = KFold(n_splits=CV_FOLD)

'''
if args.target_group == 'four':
    train_data = joblib.load('data/four_member_only/multi_gaze_training_stretch_four_group.pkl')
    test_data = joblib.load('data/four_member_only/multi_gaze_testing_stretch_four_group.pkl')

elif args.target_group == 'all':
    train_data = joblib.load('data/all_group/four_class/multi_gaze_training_stretch_all_group.pkl')
    test_data = joblib.load('data/all_group/four_class/multi_gaze_testing_stretch_all_group.pkl')
    
elif args.target_group == 'five':
    train_data = joblib.load('data/all_group/five_class/multi_gaze_training_stretch_std_all_group_5class.pkl')
    test_data = joblib.load('data/all_group/five_class/multi_gaze_testing_stretch_std_all_group_5class.pkl')
'''
  
train_data = joblib.load('data/all_group/four_class/multi_gaze_training_stretch_all_group.pkl')
test_data = joblib.load('data/all_group/four_class/multi_gaze_testing_stretch_all_group.pkl')

train_recording = sorted(set(train_data['recording']))
test_lst = sorted(set(test_data['recording']))
# pdb.set_trace()
FEATURE = args.features.split(',')
input_dims = len(FEATURE)
NODE = [args.node_num for i in range(args.layer_num)] + [args.output_dim]

output_path = args.output_path


col = ['gaze_angle_x', 'gaze_angle_y', 
        'head_angle_x', 'head_angle_y', 'head_angle_z',
        # 'head_pos_x', 'head_pos_y', 'head_pos_z', 
        'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
        'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
# train_data[col] = scaler.fit_transform(train_data[col])
# test_data[col] = scaler.transform(test_data[col])


#%%

for fold, (train_index, valid_index) in enumerate(skf.split(train_recording)):
    scaler = StandardScaler()
    # if fold == 1: import pdb;pdb.set_trace()
    model_path = './best_model/nn_result_cv{}.sav'.format(fold)
    
    train_lst = [train_recording[i] for i in train_index]
    valid_lst = [train_recording[i] for i in valid_index]
    
    train_temp = train_data.copy()
    test_temp = test_data.copy()
    
    train_df = train_temp[train_temp['recording'].isin(train_lst)]
    valid_df = train_temp[train_temp['recording'].isin(valid_lst)]
    test_df = test_temp[test_temp['recording'].isin(test_lst)]
    
    if args.std == 'yes':
        train_df[col] = scaler.fit_transform(train_df[FEATURE])
        valid_df[col] = scaler.transform(valid_df[FEATURE])
        test_df[col] = scaler.transform(test_df[FEATURE])
    
    # import pdb;pdb.set_trace()
    
    train_set = multi(train_df, FEATURE)
    valid_set = multi(valid_df, FEATURE)
    test_set = multi(test_df, FEATURE)
    model = DNN(input_dims, NODE, args.dropout)
    if args.cuda:
        model = model.cuda()
        DTYPE = torch.cuda.FloatTensor
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # setup training

    min_valid_loss = float('Inf')
    train_iterator = DataLoader(train_set, batch_size=batch_sz, num_workers=4, shuffle=True,  drop_last=True)
    valid_iterator = DataLoader(valid_set, batch_size=len(valid_set), num_workers=4, shuffle=True)
    test_iterator = DataLoader(test_set, batch_size=len(test_set), num_workers=4, shuffle=True)
    
    for e in range(args.epochs):
        model.train()
        model.zero_grad()
        avg_train_loss = 0.0
        for batch in train_iterator:
            model.zero_grad()
            # import pdb; pdb.set_trace()
            x = Variable(batch[0].float().type(DTYPE), requires_grad=False)
            y = Variable(batch[1].float().type(torch.cuda.LongTensor), requires_grad=False)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            avg_loss = loss.data
            avg_train_loss += avg_loss / len(train_set)
            optimizer.step()
    
        # print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))
    
        # Terminate the training process if run into NaN
        if np.isnan(avg_train_loss.cpu()):
            print("Training got into NaN values...\n\n")
            pdb.set_trace()
    
        model.eval()
        for batch in valid_iterator:
            x = Variable(batch[0].float().type(DTYPE), requires_grad=False)
            y = Variable(batch[1].float().type(torch.cuda.LongTensor), requires_grad=False)
            output = model(x)
            valid_loss = criterion(output, y)
            avg_valid_loss = valid_loss.data

        if np.isnan(avg_valid_loss.cpu()):
            print("Validation got into NaN values...\n\n")
            pdb.set_trace()

        avg_valid_loss = avg_valid_loss / len(valid_set)
        # print("Validation loss is: {}".format(avg_valid_loss))
    
        if (avg_valid_loss < min_valid_loss):
            min_valid_loss = avg_valid_loss
            torch.save(model, model_path)
            # print("Found new best model, saving to disk...")

        # print("\n\n")
    
    
    model = torch.load(model_path)
    model.eval()
    for batch in test_iterator:
        x = Variable(batch[0].float().type(DTYPE), requires_grad=False)
        y = Variable(batch[1].float().type(torch.cuda.LongTensor), requires_grad=False)
        output_test = model(x)
        loss_test = criterion(output_test, y)
        avg_test_loss = loss_test.data / len(test_set)
        output_test = output_test.cpu().data.numpy().reshape(-1, args.output_dim)
        y = y.cpu().data.numpy()
    
        # these are the needed metrics
        output_test = np.argmax(output_test, axis=-1)

        bi_acc = accuracy_score(y, output_test)
        f1 = f1_score(y, output_test, average='macro')
        precision = precision_score(y, output_test, average='macro', zero_division=0)
        recall = recall_score(y, output_test, average='macro')
        uar = (precision+recall)/2.0
        print('learning rate: {}  layer_num: {}  node_num: {}'.format(LR, args.layer_num, args.node_num))
        display(bi_acc, f1, recall, precision, uar)
        print('='*80)
        print('\n')
    
        with open(output_path, 'a') as out:
            writer = csv.writer(out)
            writer.writerow([fold, args.std, LR, args.batch_size, args.layer_num, args.node_num, args.dropout, args.epochs, ' + '.join(FEATURE), bi_acc, f1, recall, precision, uar])

