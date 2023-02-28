#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:41:14 2022

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

# import pdb;pdb.set_trace()
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
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
sample_df = sample_df.reset_index(drop=True)
sample_df = sample_df.apply(turn_frame, axis=1)


#%% talk activaty
recording_set = sorted(list(set(anno_df['recording'])))
cal = pd.DataFrame()
talk_df = pd.DataFrame()
silence=[]
talking=[]
silence_count = pd.Series()
talking_count = pd.Series()
for r in recording_set:
    temp_annotation = anno_df[anno_df['recording']==r].T
    new_header = list(temp_annotation.loc['subject'])
    temp_annotation = temp_annotation[2:]
    temp_annotation.columns = new_header
    temp_annotation['subjectPos1'] = temp_annotation['subjectPos1'].apply(lambda x: 't' if x==1 else 's')
    temp_annotation['subjectPos2'] = temp_annotation['subjectPos2'].apply(lambda x: 't' if x==1 else 's')
    temp_annotation['subjectPos3'] = temp_annotation['subjectPos3'].apply(lambda x: 't' if x==1 else 's')
    temp_annotation['subjectPos4'] = temp_annotation['subjectPos4'].apply(lambda x: 't' if x==1 else 's')
    temp_annotation = temp_annotation.reset_index(drop=True)
    temp_annotation.insert(loc=0, column='recording', value=[r for i in range(len(temp_annotation))])
    
    for ids in range(1,5):
        ac = temp_annotation.copy()
        ac['subgroup{}'.format(ids)] = (temp_annotation['subjectPos{}'.format(ids)] != temp_annotation['subjectPos{}'.format(ids)].shift(1)).cumsum()
        ac = ac.groupby('subgroup{}'.format(ids),as_index=False).apply(lambda x: x['subjectPos{}'.format(ids)].head(1)).reset_index(drop=True)
        ac = ac.drop(columns=['subgroup{}'.format(ids)])
        if len(ac)==1:
                continue
        
        count = temp_annotation.copy()
        count['subgroup{}'.format(ids)] = (temp_annotation['subjectPos{}'.format(ids)] != temp_annotation['subjectPos{}'.format(ids)].shift(1)).cumsum()
        count = count.groupby('subgroup{}'.format(ids),as_index=False).apply(lambda x: x.shape[0])
        count = count.rename(columns={None:'len'})
        count=count.drop(columns=['subgroup{}'.format(ids)])
        
        # ac = ac.rename(columns={})
        temp_stat = pd.concat([ac, count], axis=1)
        mean_stat = temp_stat.groupby('subjectPos{}'.format(ids)).mean()
        silence_count_temp = temp_stat[temp_stat['subjectPos{}'.format(ids)]=='s']['len']
        silence_count_temp = silence_count_temp.drop([0, silence_count_temp.index[-1]])
        silence_count = silence_count.append(silence_count_temp)
        
        talking_count_temp = temp_stat[temp_stat['subjectPos{}'.format(ids)]=='t']['len']
        talking_count = talking_count.append(talking_count_temp)
        
        silence.append(mean_stat.loc['s', 'len'])
        talking.append(mean_stat.loc['t', 'len'])
        # import pdb;pdb.set_trace()
print('silence length: {}, std: {}'.format(np.array(silence).mean(), np.array(silence).std()))
print('talk length: {}, std: {}'.format(np.array(talking).mean(), np.array(talking).std()))
silence_count.value_counts().sort_index().plot()
talking_count.value_counts().sort_index().plot()
pdb.set_trace()

#%%
annotation_file = "/homes/GPU2/shaohao/Corpus/multimediate/labels/next_speaker_annotation.csv"
anno_df = pd.read_csv(annotation_file, index_col=0)   
anno_df = anno_df.T
anno_df = anno_df.reset_index()
anno_df = anno_df.rename(columns={'index': 'recording', anno_df.columns[1]: 'subject'})
anno_df['recording'] = anno_df['recording'].apply(lambda x: x.split('.')[0])

recording_set = sorted(list(set(anno_df['recording'])))
cal = pd.DataFrame()
talk_df = pd.DataFrame()
silence=[]
talking=[]
silence_count = pd.Series()
talking_count = pd.Series()
for r in recording_set:
    temp_annotation = anno_df[anno_df['recording']==r].T
    new_header = list(temp_annotation.loc['subject'])
    temp_annotation = temp_annotation[2:]
    temp_annotation.columns = new_header
    temp_annotation = temp_annotation.reset_index(drop=True)
    temp_annotation = temp_annotation.dropna(how='all')
    temp_annotation = temp_annotation.fillna(0)
    temp_annotation['subjectPos1'] = pd.to_numeric(temp_annotation['subjectPos1'])
    temp_annotation['subjectPos2'] = pd.to_numeric(temp_annotation['subjectPos2'])
    temp_annotation['subjectPos3'] = pd.to_numeric(temp_annotation['subjectPos3'])
    temp_annotation['subjectPos4'] = pd.to_numeric(temp_annotation['subjectPos4'])
    temp_annotation['sum']=temp_annotation[['subjectPos1', 'subjectPos2', 'subjectPos3', 'subjectPos4']].sum(axis=1)
    temp_annotation['sum'] = temp_annotation['sum'].apply(lambda x:'t' if x>0 else 's')
    
    import pdb;pdb.set_trace()
    ac = temp_annotation.copy()
    ac['subgroup'] = (temp_annotation['sum'] != temp_annotation['sum'].shift(1)).cumsum()
    ac = ac.groupby('subgroup',as_index=False).apply(lambda x: x['sum'.format(ids)].head(1)).reset_index(drop=True)
    ac = ac.drop(columns=['subgroup'])
    if len(ac)==1:
            continue
    # pdb.set_trace()
    count = temp_annotation.copy()
    count['subgroup'] = (temp_annotation['sum'] != temp_annotation['sum'].shift(1)).cumsum()
    count = count.groupby('subgroup',as_index=False).apply(lambda x: x.shape[0])
    count = count.rename(columns={None:'len'})
    count=count.drop(columns=['subgroup'])
    
    # ac = ac.rename(columns={})
    temp_stat = pd.concat([ac, count], axis=1)
    mean_stat = temp_stat.groupby('sum').mean()
    silence_count_temp = temp_stat[temp_stat['sum']=='s']['len']
    silence_count = silence_count.append(silence_count_temp)
    
    talking_count_temp = temp_stat[temp_stat['sum']=='t']['len']
    talking_count = talking_count.append(talking_count_temp)
        
    silence.append(mean_stat.loc['s', 'len'])
    talking.append(mean_stat.loc['t', 'len'])
        # import pdb;pdb.set_trace()
print('silence length: {}, std: {}'.format(np.array(silence).mean(), np.array(silence).std()))
print('talk length: {}, std: {}'.format(np.array(talking).mean(), np.array(talking).std()))
silence_count.value_counts().sort_index().plot()
talking_count.value_counts().sort_index().plot()
pdb.set_trace()
#%% interrupt
annotation_file = "/homes/GPU2/shaohao/Corpus/multimediate/labels/next_speaker_annotation.csv"
anno_df = pd.read_csv(annotation_file, index_col=0)   
anno_df = anno_df.T
anno_df = anno_df.reset_index()
anno_df = anno_df.rename(columns={'index': 'recording', anno_df.columns[1]: 'subject'})
anno_df['recording'] = anno_df['recording'].apply(lambda x: x.split('.')[0])

recording_set = sorted(list(set(anno_df['recording'])))
cal = pd.DataFrame()
talk_df = pd.DataFrame()
silence=[]
talking=[]
silence_count = pd.Series()
talking_count = pd.Series()
interrupt_lst = []
for i in range(30):
    interrupt=0
    for r in tqdm.tqdm(recording_set):
        temp_annotation = anno_df[anno_df['recording']==r].T
        new_header = list(temp_annotation.loc['subject'])
        temp_annotation = temp_annotation[2:]
        temp_annotation.columns = new_header
        temp_annotation = temp_annotation.reset_index(drop=True)
        temp_annotation = temp_annotation.dropna(how='all')
        temp_annotation = temp_annotation.fillna(0)
        temp_annotation['subjectPos1'] = pd.to_numeric(temp_annotation['subjectPos1'])
        temp_annotation['subjectPos2'] = pd.to_numeric(temp_annotation['subjectPos2'])
        temp_annotation['subjectPos3'] = pd.to_numeric(temp_annotation['subjectPos3'])
        temp_annotation['subjectPos4'] = pd.to_numeric(temp_annotation['subjectPos4'])
        
        temp_sample = sample_df.copy()
        temp_sample = temp_sample[temp_sample['recording']==r].reset_index(drop=True)
        for ind, row in temp_sample.iterrows():
            if (temp_annotation.loc[row['end_time']-1,:]==temp_annotation.loc[row['end_time']+i, :]).all():
                # print(temp_annotation.loc[row['end_time']-1,:])
                # print(temp_annotation.loc[row['end_time']+30,:])
                # print('='*30)
                interrupt+=1
    interrupt_lst.append(interrupt)
# pdb.set_trace()
import matplotlib.pyplot as plt

plt.plot([i for i in range(30)],interrupt_lst)
plt.title('plot')
plt.xlabel('g8\f ```5`][;36ap')
plt.ylabel('same')
plt.show()
    
 #%%
recording_set = sorted(list(set(anno_df['recording'])))
weight = [i/299 for i in range(300)]

talk_df = pd.DataFrame()
for r in recording_set:
    temp_annotation = anno_df[anno_df['recording']==r].T
    new_header = list(temp_annotation.loc['subject'])
    temp_annotation = temp_annotation[2:]
    temp_annotation.columns = new_header
    temp_annotation['subjectPos1'] = pd.to_numeric(temp_annotation['subjectPos1'])
    temp_annotation['subjectPos2'] = pd.to_numeric(temp_annotation['subjectPos2'])
    temp_annotation['subjectPos3'] = pd.to_numeric(temp_annotation['subjectPos3'])
    temp_annotation['subjectPos4'] = pd.to_numeric(temp_annotation['subjectPos4'])
   
    temp_sample = sample_df[sample_df['recording']==r]
    
    for ind, row in temp_sample.iterrows():
        cur_df = temp_annotation.iloc[row['start_time']:row['end_time']+30]
        import pdb;pdb.set_trace()
        cur_df= cur_df.sum()
        temp = pd.concat([row, cur_df], axis=0)
        talk_df = pd.concat([talk_df, temp], axis=1)
        # pdb.set_trace()
    
    
talk_df = talk_df.T.reset_index(drop=True)
talk_df = talk_df[['recording']+list(talk_df.columns[-8:])]
talk_df[list(talk_df.columns[-8:])] = talk_df[list(talk_df.columns[-8:])].apply(pd.to_numeric)
talk_df = talk_df.apply(to_ratio, axis=1)
    



