#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:32:19 2022

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

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str,  dest='file', default="path~~",help='output file')
parser.add_argument('--feature', type=str,  dest='feature', default='gaze_angle_x,gaze_angle_y,head_angle_x,head_angle_y,head_angle_z,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_1_z',help='output file')
parser.add_argument('--model', type=str,  dest='model', default="svm",help='svm randomforest')
parser.add_argument('--seed', type=int,  dest='seed', default=1,help='seed range')

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

#%%
seed = args.seed
np.random.seed(seed)
random.seed(seed)
CV_FOLD = 5
skf = KFold(n_splits=CV_FOLD)
feature = args.feature.split(',')
train_df = joblib.load("data/all_group/four_class/multi_gaze_training_stretch_all_group.pkl")
test_df = joblib.load("data/all_group/four_class/multi_gaze_testing_stretch_all_group.pkl")

recording_set = sorted(set(train_df.recording))

Test, Test_y = get_data_y_multi(test_df, sorted(set(test_df.recording)), feature)
#%%
total_f1 = []
total_acc = []
total_recall = []
total_precision = []
total_uar = []
for seed in range(args.seed):
    np.random.seed(seed)
    random.seed(seed)
    TEST_ground_truth = []
    TEST_pred = []
         
    for fold, (train_index, valid_index) in tqdm.tqdm(enumerate(skf.split(recording_set))):
        ground_truth = []
        pred = []
        origin_uar = -1
        # import pdb;pdb.set_trace()
        
        train_lst = [recording_set[i] for i in train_index]
        val_lst = [recording_set[i] for i in valid_index]

        Train, Train_y = get_data_y_multi(train_df, train_lst, feature)
        Valid, Valid_y = get_data_y_multi(train_df, val_lst, feature)
             
        model = SVC(gamma='auto')
        
        model.fit(Train, Train_y)
        y_pred = model.predict(Valid)
        
        pred.extend(list(y_pred))
        ground_truth.extend(list(Valid_y))
   
        f1score = f1_score(ground_truth,pred, average='macro')
        acc=accuracy_score(ground_truth,pred)
        precision = precision_score(ground_truth, pred, average='macro')
        recall = recall_score(ground_truth, pred, average='macro')
        uar = (precision+recall)/2.0
        
        if uar >= origin_uar:
            origin_uar = uar
            filename = './best_model/svm_cv{}.sav'.format(fold)
            pickle.dump(model, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))

    y_pred = loaded_model.predict(Test)

    TEST_pred.extend(list(y_pred))
    TEST_ground_truth.extend(list(Test_y))
    
    TEST_f1score = f1_score(TEST_ground_truth,TEST_pred, average='macro', zero_division=0)
    TEST_acc=accuracy_score(TEST_ground_truth,TEST_pred)
    TEST_precision = precision_score(TEST_ground_truth, TEST_pred, average='macro', zero_division=0)
    TEST_recall = recall_score(TEST_ground_truth, TEST_pred, average='macro', zero_division=0)
    TEST_uar = (TEST_precision+TEST_recall)/2.0
        
    total_f1.append(TEST_f1score)
    total_acc.append(TEST_acc)
    total_uar.append(TEST_uar)
    total_recall.append(TEST_recall)
    total_precision.append(TEST_precision)
    
    print('SEED: {}, f1 score: {:.4f}, acc: {:.4f}, recall: {:.4f}, UAR: {:.4f}'.format(seed, TEST_f1score, TEST_acc, TEST_recall, TEST_uar))
    # with open(args.file, 'a', newline = '') as c:
    #     writer = csv.writer(c)
    #     writer.writerow([seed, CV_FOLD, TEST_acc, TEST_f1score, TEST_uar])
    
print('Mean f1 score: {:.4f}, mean acc: {:.4f}, mean recall: {:.4f}, mean uar: {:.4f}'.format(np.mean(total_f1),
                                                                                                       np.mean(total_acc),
                                                                                                       np.mean(total_recall),
                                                                                                       np.mean(total_uar)))

with open(args.file, 'a') as out:
    writer = csv.writer(out)
    writer.writerow([' + '.join(feature), np.mean(total_acc), np.mean(total_f1), np.mean(total_recall), np.mean(total_precision), np.mean(total_uar)])