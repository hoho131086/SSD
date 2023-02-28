#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 00:40:12 2022

@author: shaohao
"""

import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import random
import tqdm
import argparse
import csv
import pdb
import math
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str,  dest='file', default="path~~",help='output file')
parser.add_argument('--feature', type=str,  dest='feature', default='angle_x,angle_y',help='output file')
parser.add_argument('--model', type=str,  dest='model', default="svm",help='svm randomforest')
parser.add_argument('--seed', type=int,  dest='seed', default=10,help='seed range')

args = parser.parse_args()
# pdb.set_trace()
#%%
def get_data_y_multi(df, lst, feature):
    # pdb.set_trace()
    clear = df[df['recording'].isin(lst)]
    # Data = clear[['angle_x', 'angle_y']].values  
    Data = clear[feature]
    y = np.vstack(clear['label'])
    
    return Data.astype(np.float), y.ravel()

def find_nearest_dir(value):
    array = np.array([math.radians(-45), math.radians(0), math.radians(45),  math.radians(90), math.radians(-90)])
    direction = [1, 2, 3, 0, 0]
    idx = (np.abs(array - value)).argmin()
    return direction[idx]
    
#%%

train_df = joblib.load("data/all_group/four_class/multi_gaze_training_stretch_all_group.pkl")
test_df = joblib.load("data/all_group/four_class/multi_gaze_testing_stretch_all_group.pkl")
data = pd.concat([train_df, test_df], axis=0)
# pdb.set_trace()
pred = test_df['gaze_angle_x'].apply(find_nearest_dir).values
target = test_df['label'].values.astype(int)

    
TEST_f1score = f1_score(target,pred, average='macro', zero_division=0)
TEST_acc=accuracy_score(target,pred)
TEST_precision = precision_score(target, pred, average='macro', zero_division=0)
TEST_recall = recall_score(target, pred, average='macro', zero_division=0)
TEST_uar = (TEST_precision+TEST_recall)/2.0


print('acc: {:.4f}, f1 score: {:.4f}, recall: {:.4f}, precision: {:.4f}, UAR: {:.4f}'.format(TEST_acc, TEST_f1score, TEST_recall, TEST_uar, TEST_precision))

train_df = joblib.load("data/all_group/four_class/multi_gaze_training_stretch_all_group.pkl")
test_df = joblib.load("data/all_group/four_class/multi_gaze_testing_stretch_all_group.pkl")
data = pd.concat([train_df, test_df], axis=0)
# pdb.set_trace()
pred = test_df['head_angle_x'].apply(find_nearest_dir).values
target = test_df['label'].values.astype(int)

    
TEST_f1score = f1_score(target,pred, average='macro', zero_division=0)
TEST_acc=accuracy_score(target,pred)
TEST_precision = precision_score(target, pred, average='macro', zero_division=0)
TEST_recall = recall_score(target, pred, average='macro', zero_division=0)
TEST_uar = (TEST_precision+TEST_recall)/2.0


print('acc: {:.4f}, f1 score: {:.4f}, recall: {:.4f}, precision: {:.4f}, UAR: {:.4f}'.format(TEST_acc, TEST_f1score, TEST_recall, TEST_uar, TEST_precision))
