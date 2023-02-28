#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:25:09 2022

@author: shaohao
"""
import random
import pdb
import pandas as pd
import numpy as np

import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import tqdm
import joblib

import warnings
warnings.simplefilter("ignore")

def turn_frame(df):
    df['start_time'] = int(np.round(df['start_time']/(1/30)))
    df['end_time'] = int(np.round(df['end_time']/(1/30)))
    return df

def to_ratio(df):
    temp_sum = 300
    df['subjectPos1'] = df['subjectPos1']/temp_sum
    df['subjectPos2'] = df['subjectPos2']/temp_sum
    df['subjectPos3'] = df['subjectPos3']/temp_sum
    df['subjectPos4'] = df['subjectPos4']/temp_sum
    return df

def map_label(df):
    return 8*df['label_1']+4*df['label_2']+ 2*df['label_3']+df['label_4']


def cos_sim(df):
    label_col = list(df.index)[1:5]
    anno_col = list(df.index)[-4:]
    # pdb.set_trace()
    return 1 - scipy.spatial.distance.cosine(df[label_col].values.astype(np.float), df[anno_col].values.astype(np.float))

def pearson(df):
    label_col = list(df.index)[1:5]
    anno_col = list(df.index)[-4:]
    # pdb.set_trace()
    return scipy.stats.pearsonr(df[label_col].values.astype(np.float), df[anno_col].values.astype(np.float))


#%% data preprocess
annotation_file = "/homes/GPU2/shaohao/Corpus/multimediate/labels/next_speaker_annotation.csv"
anno_df = pd.read_csv(annotation_file, index_col=0)   
anno_df = anno_df.T
anno_df = anno_df.reset_index()
anno_df = anno_df.rename(columns={'index': 'recording', anno_df.columns[1]: 'subject'})
anno_df['recording'] = anno_df['recording'].apply(lambda x: x.split('.')[0])
anno_df = anno_df.fillna(0)

sample_df = pd.DataFrame()
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
train_lst = sorted(set(temp_sample['recording']))
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
sample_df = sample_df.reset_index(drop=True)
sample_df = sample_df.apply(turn_frame, axis=1)
test_lst = sorted(set(temp_sample['recording']))
# pdb.set_trace()
all_lst = train_lst + test_lst
#%% talk times
'''
recording_set = sorted(list(set(anno_df['recording'])))

talk_df = pd.DataFrame()
for r in recording_set:
    temp_annotation = anno_df.copy()
    temp_annotation = temp_annotation[temp_annotation['recording']==r].T
    new_header = temp_annotation.loc['subject']
    temp_annotation = temp_annotation[2:]
    temp_annotation.columns = new_header
    temp_annotation['subjectPos1'] = pd.to_numeric(temp_annotation['subjectPos1'])
    temp_annotation['subjectPos2'] = pd.to_numeric(temp_annotation['subjectPos2'])
    temp_annotation['subjectPos3'] = pd.to_numeric(temp_annotation['subjectPos3'])
    temp_annotation['subjectPos4'] = pd.to_numeric(temp_annotation['subjectPos4'])
   
    temp_sample = sample_df.copy()
    temp_sample = temp_sample[temp_sample['recording']==r]
    
    for ind, row in temp_sample.iterrows():
        cur_df = temp_annotation.copy()
        cur_df = cur_df.iloc[row['start_time']:row['end_time']]
        cur_df= cur_df.sum()
        temp = pd.concat([row, cur_df], axis=0)
        talk_df = pd.concat([talk_df, temp], axis=1)
        # pdb.set_trace()
    
    
talk_df = talk_df.T.reset_index(drop=True)
talk_df = talk_df[['recording']+list(talk_df.columns[-8:])]
talk_df[list(talk_df.columns[-8:])] = talk_df[list(talk_df.columns[-8:])].apply(pd.to_numeric)
talk_df = talk_df.apply(to_ratio, axis=1)
talk_df.to_csv('result_csv/talk_tend.csv')
'''
# import pdb;pdb.set_trace()
#%% correlation + similarity
talk_df = pd.read_csv('result_csv/talk_tend.csv', index_col=0)
trans_dic = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/final/DED/idx/transit.pkl")
# pdb.set_trace()
# talk_df = talk_df[talk_df['recording'].isin(test_lst)]
# output_sim = talk_df.apply(cos_sim, axis=1)
# output_sim = output_sim.dropna()

# pear = talk_df.apply(pearson, axis=1)
# pear = pd.DataFrame(pear, columns=['tuple'])
# pear['correlation'], pear['P_value'] = zip(*pear['tuple'])
# pear = pear.dropna()
# pear = pear.drop(columns=['tuple'])

# print('cosine sim: {:.4f}'.format(output_sim.mean()))
# print(pear.mean())
#%%

print('old')
def thres(df, th):
    if df['subjectPos1'] >= th:
        df['subjectPos1'] = 1
    else:
        df['subjectPos1'] = 0
    
    if df['subjectPos2'] >= th:
        df['subjectPos2'] = 1
    else:
        df['subjectPos2'] = 0
        
    if df['subjectPos3'] >= th:
        df['subjectPos3'] = 1
    else:
        df['subjectPos3'] = 0
        
    if df['subjectPos4'] >= th:
        df['subjectPos4'] = 1
    else:
        df['subjectPos4'] = 0
        
    return df

precision_lst=[]
f1_lst=[]
uar_lst=[]
acc_lst = []
thresh_var = 30
statistic = pd.DataFrame()


for thresh_var in range(0,105,10):
    gt = []
    pred = []
    for ind, recording in enumerate(test_lst):
        thresh_df = talk_df.copy()
        thresh_df = thresh_df[thresh_df['recording']==recording]
        thresh_df = thresh_df.apply(thres, th = (thresh_var/100.0), axis=1)
        # pdb.set_trace()
        for member in range(1,5):
            # gt = np.hstack([thresh_df['label_1'].values, thresh_df['label_2'].values, 
            #                 thresh_df['label_3'].values, thresh_df['label_4'].values])
            # pred =  np.hstack([thresh_df['subjectPos1'].values, thresh_df['subjectPos2'].values, 
            #                 thresh_df['subjectPos3'].values, thresh_df['subjectPos4'].values])
            gt.extend(list(thresh_df['label_{}'.format(member)]))
            pred.extend(list(thresh_df['subjectPos{}'.format(member)]))
            
            # precision = precision_score(gt, pred, average='macro')
            # recall = recall_score(gt, pred, average='macro')
            # uar = (precision+recall)/2.0
            # acc = accuracy_score(gt, pred)
            
            # statistic.loc[ind, 'recording'] = recording
            # statistic.loc[ind, 'member{}'.format(member)] = uar
    
            
    precision = precision_score(gt, pred, average='macro')
    uar = recall_score(gt, pred, average='macro')
    f1 = f1_score(gt, pred, average = 'macro')
    acc = accuracy_score(gt, pred)
    precision_lst.append(precision)
    f1_lst.append(f1)
    uar_lst.append(uar)
    acc_lst.append(acc)
    print('thresh hold: {:.2f}, precision: {:.4f}, f1: {:.4f}, uar: {:.4f}, acc: {:.4f}'
          .format(thresh_var/100.0, precision, f1, uar, acc))

result = pd.DataFrame({'precision': precision_lst, 'f1':f1_lst, 'uar': uar_lst, 'acc': acc_lst})
result.to_csv('result_csv/talk_result_multi_all.csv')
# %%
'''
from sklearn.preprocessing import Normalizer
import math


def get_weighted(length):
    return [(i+1)*(1.0/(length+1)) for i in range(length+1)]

def get_exp(length):
    return [math.exp(i-length) for i in range(length+1)]

def get_one(length):
    return [1 for i in range(length+1)]

method = 'exp'
print('Try {}'.format(method))

col_output = ['label_1', 'label_2', 'label_3', 'label_4', 'subjectPos1', 'subjectPos2', 'subjectPos3', 'subjectPos4']
col = ['subjectPos1', 'subjectPos2', 'subjectPos3', 'subjectPos4']

output = pd.DataFrame(columns=col_output)
for recording in sorted(set(talk_df['recording'])):
    recording_df = talk_df.copy()
    recording_df = recording_df[recording_df['recording']==recording]
    recording_df = recording_df.reset_index(drop=True)
    
    for ind, row in recording_df.iterrows():
        recording_df_copy = recording_df.copy()
        temp_df = recording_df_copy.loc[:ind, col]
        if ind < 4:
            LEN = ind
        else: LEN = 3
        if method == 'weight':
            w = get_weighted(LEN)
        elif method == 'exp':
            w = get_exp(LEN)    
        elif method=='average':
            w = get_one(LEN)
            
        for member in range(1,5):
            if ind < 4:
                temp_df.loc[ind, 'subjectPos{}'.format(member)] = temp_df.loc[:ind,'subjectPos{}'.format(member)]*w
            else:
                temp_df.loc[ind, 'subjectPos{}'.format(member)] = temp_df.loc[ind-3:ind,'subjectPos{}'.format(member)]*w
        # temp_df = temp_df.mean()
        temp_df = Normalizer(norm='l2').fit_transform(temp_df.sum().values.reshape(1,-1))
        temp_df = pd.Series(temp_df.squeeze()).rename(index={0:'subjectPos1', 1:'subjectPos2',2:'subjectPos3',3:'subjectPos4'})
        temp_result = temp_df.append(recording_df_copy.loc[ind, ['label_1', 'label_2', 'label_3', 'label_4']])
        # pdb.set_trace()
        output = output.append(temp_result, ignore_index=True)

# pdb.set_trace()

for thresh_var in range(0, 100, 5):
    
    thresh_df = output.copy()
    thresh_df = thresh_df.apply(thres, th = (thresh_var/100.0), axis=1)
    gt = np.hstack([thresh_df['label_1'].values, thresh_df['label_2'].values, 
                    thresh_df['label_3'].values, thresh_df['label_4'].values])
    pred =  np.hstack([thresh_df['subjectPos1'].values, thresh_df['subjectPos2'].values, 
                    thresh_df['subjectPos3'].values, thresh_df['subjectPos4'].values])
    
    precision = precision_score(gt, pred, average='macro')
    recall = recall_score(gt, pred, average='macro')
    uar = (precision+recall)/2.0
    precision_lst.append(precision)
    recall_lst.append(recall)
    uar_lst.append(uar)
    print('thresh hold: {:.2f}, precision: {:.4f}, recall: {:.4f}, uar: {:.4f}'.format(thresh_var/100.0, precision, recall, uar)) 

'''
        
#%% model prediction

'''
#%%
def get_data_y_multi(index, df, lst):

    clear = df[df['recording'].isin(lst)]
    clear['input'] = clear.apply(lambda x: x[list(clear.columns[-4:])].values, axis=1)
    Data = np.vstack(clear['input'])
    # pdb.set_trace()
    y_1 = np.vstack(clear['label_1'])
    y_2 = np.vstack(clear['label_2'])
    y_3 = np.vstack(clear['label_3'])
    y_4 = np.vstack(clear['label_3'])
    return Data.astype(np.float), y_1.ravel(), y_2.ravel(), y_3.ravel(), y_4.ravel()

#%%
SEED = 10
CV_FOLD = 5
skf = KFold(n_splits=CV_FOLD)
unique = talk_df.drop_duplicates(subset=['recording'])

total_f1 = []
total_acc = []
total_uar = []
for seed in range(SEED):

    np.random.seed(seed)
    random.seed(seed)
    ground_truth = []
    pred = []
    
    for fold, (train_index, test_index) in tqdm.tqdm(enumerate(skf.split(unique))):
      
        model_1 = RandomForestClassifier()
        model_2 = RandomForestClassifier()
        model_3 = RandomForestClassifier()
        model_4 = RandomForestClassifier()
        
        train_lst = list(unique.iloc[train_index]['recording'])
        test_lst = list(unique.iloc[test_index]['recording'])
        
        Train , Train_y_1, Train_y_2, Train_y_3, Train_y_4 = get_data_y_multi(train_index, talk_df, train_lst)
        Test  , Test_y_1, Test_y_2, Test_y_3, Test_y_4 = get_data_y_multi(test_index, talk_df, test_lst)
        
        sc = StandardScaler()
        sc.fit(Train)
        Train = sc.transform(Train)
        Test = sc.transform(Test)
        
        model_1.fit(Train, Train_y_1)
        model_2.fit(Train, Train_y_2)
        model_3.fit(Train, Train_y_3)
        model_4.fit(Train, Train_y_4)
        y_pred_1 = model_1.predict(Test)
        y_pred_2 = model_2.predict(Test)
        y_pred_3 = model_3.predict(Test)
        y_pred_4 = model_4.predict(Test)
        
        pred.extend(list(y_pred_1))
        pred.extend(list(y_pred_2))
        pred.extend(list(y_pred_3))
        pred.extend(list(y_pred_4))
        ground_truth.extend(list(Test_y_1))
        ground_truth.extend(list(Test_y_2))
        ground_truth.extend(list(Test_y_3))
        ground_truth.extend(list(Test_y_4))
             
    f1score = f1_score(ground_truth,pred, average='macro')
    acc=accuracy_score(ground_truth,pred)
    precision = precision_score(ground_truth, pred, average='macro')
    recall = recall_score(ground_truth, pred, average='macro')
    uar = (precision+recall)/2.0
    print("Validation UAR: {}".format(uar))
    
    total_f1.append(f1score)
    total_acc.append(acc)
    total_uar.append(uar)
    
    print('SEED: {}, f1 score: {:.4f}, acc: {:.4f}, UAR: {:.4f}'.format(seed, f1score, acc, uar))


'''







