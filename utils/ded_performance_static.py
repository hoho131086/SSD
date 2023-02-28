#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:37:40 2022

@author: shaohao
"""
import pandas as pd
import pdb
import numpy as np
import tqdm

sample_df = pd.DataFrame()
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
sample_df = sample_df.reset_index(drop=True)

annotation_file = "/homes/GPU2/shaohao/Corpus/multimediate/labels/next_speaker_annotation.csv"
anno_df = pd.read_csv(annotation_file, index_col=0)   
anno_df = anno_df.T
anno_df = anno_df.reset_index()
anno_df = anno_df.rename(columns={'index': 'recording', anno_df.columns[1]: 'subject'})
anno_df['recording'] = anno_df['recording'].apply(lambda x: x.split('.')[0])
anno_df['subject'] = anno_df['subject'].apply(lambda x: int(x.split('Pos')[1]))

recording_set = sorted(list(set(anno_df['recording'])))
# recording_set = ['recording12']
# pdb.set_trace()
#%%
member_duration_mean = pd.DataFrame()
member_duration_std = pd.DataFrame()
for ind, recording in tqdm.tqdm(enumerate(recording_set)):
    member_duration_temp = pd.DataFrame(columns=['member1_mean', 'member2_mean', 'member3_mean', 'member4_mean',
                                                 'member1_sum', 'member2_sum', 'member3_sum', 'member4_sum',
                                                 'member1_max', 'member2_max', 'member3_max', 'member4_max'])
    temp_anno = anno_df.copy()
    temp_anno = temp_anno[temp_anno['recording']==recording].T
    temp_anno = temp_anno.rename(columns=temp_anno.iloc[1])
    temp_anno = temp_anno.iloc[2:]
    temp_anno = temp_anno.dropna(axis = 0, how = 'all')
    lack_member = temp_anno.columns[temp_anno.isna().all()].tolist()
    temp_anno = temp_anno.fillna(0)
    temp_anno = temp_anno.reset_index(drop=True)
    temp_anno = temp_anno.astype(float)
    temp_anno = temp_anno.astype(int)
    
    temp_sample = sample_df.copy()
    temp_sample = temp_sample[temp_sample['recording']==recording].reset_index(drop=True)
    temp_sample['start_time'] = temp_sample['start_time'].apply(lambda x: int(np.round(x*30)))
    temp_sample['end_time'] = temp_sample['end_time'].apply(lambda x: int(np.round(x*30)))
    
    for i, row in temp_sample.iterrows():
        anno_copy = temp_anno.copy()
        anno_copy = anno_copy[row['start_time']:row['end_time']]
        
        for member in range(1,5):
            
            member_talk_index = anno_copy.loc[anno_copy[member]==1].reset_index().loc[lambda x: x['index']!=x['index'].rolling(3, center=True).mean()]['index']
            
            if len(member_talk_index)!=0:
                if len(member_talk_index)%2==1:
                    member_talk_index = member_talk_index.append(pd.Series([member_talk_index.values[-1]+1]))
                member_talk = pd.DataFrame(data={'start': member_talk_index[::2].values, 'end': member_talk_index[1::2].values})
                member_talk['duration'] = member_talk.apply(lambda x: x['end'] - x['start'], axis=1)
                member_duration_temp.loc[i, 'member{}_sum'.format(member)] = member_talk['duration'].sum()
                member_duration_temp.loc[i, 'member{}_mean'.format(member)] = member_talk['duration'].mean()
                member_duration_temp.loc[i, 'member{}_max'.format(member)] = member_talk['duration'].max()
            else:
                member_duration_temp.loc[i, 'member{}_sum'.format(member)] = 0
                member_duration_temp.loc[i, 'member{}_mean'.format(member)] = 0
                member_duration_temp.loc[i, 'member{}_max'.format(member)] = 0
    member_duration_temp = member_duration_temp/30.0
    member_duration_mean = pd.concat([member_duration_mean, member_duration_temp.mean()], axis=1)
    member_duration_std = pd.concat([member_duration_std, member_duration_temp.std()], axis=1)
    # pdb.set_trace()
member_duration_mean = member_duration_mean.T.reset_index(drop=True)
member_duration_std = member_duration_std.T.reset_index(drop=True)
    
pdb.set_trace()
member_duration_mean.to_csv('in_time_duration_mean.csv')
member_duration_std.to_csv('in_time_duration_std.csv')
#%%

member_duration = pd.DataFrame(columns=['recording', 'member1_mean', 'member2_mean', 'member3_mean', 'member4_mean',
                                        'member1_sum', 'member2_sum', 'member3_sum', 'member4_sum',
                                        'member1_max', 'member2_max', 'member3_max', 'member4_max'])
for ind, recording in tqdm.tqdm(enumerate(recording_set)):
    member_duration.loc[ind, 'recording'] = recording
    temp_anno = anno_df.copy()
    temp_anno = temp_anno[temp_anno['recording']==recording].T
    temp_anno = temp_anno.rename(columns=temp_anno.iloc[1])
    temp_anno = temp_anno.iloc[2:]
    temp_anno = temp_anno.dropna(axis = 0, how = 'all')
    lack_member = temp_anno.columns[temp_anno.isna().all()].tolist()
    temp_anno = temp_anno.fillna(0)
    temp_anno = temp_anno.reset_index(drop=True)
    temp_anno = temp_anno.astype(float)
    temp_anno = temp_anno.astype(int)
    
        
    for member in range(1,5):
        anno_copy = temp_anno.copy()
        member_talk_index = anno_copy.loc[anno_copy[member]==1].reset_index().loc[lambda x: x['index']!=x['index'].rolling(3, center=True).mean()]['index']
        
        if len(member_talk_index)!=0:
            if len(member_talk_index)%2==1:
                member_talk_index = member_talk_index.append(pd.Series([member_talk_index.values[-1]+1]))
            member_talk = pd.DataFrame(data={'start': member_talk_index[::2].values, 'end': member_talk_index[1::2].values})
            member_talk['duration'] = member_talk.apply(lambda x: x['end'] - x['start'], axis=1)
            member_duration.loc[ind, 'member{}_sum'.format(member)] = member_talk['duration'].sum()
            member_duration.loc[ind, 'member{}_mean'.format(member)] = member_talk['duration'].mean()
            member_duration.loc[ind, 'member{}_max'.format(member)] = member_talk['duration'].max()
        else:
            member_duration.loc[ind, 'member{}_sum'.format(member)] = 0
            member_duration.loc[ind, 'member{}_mean'.format(member)] = 0
            member_duration.loc[ind, 'member{}_max'.format(member)] = 0
        
pdb.set_trace()
member_duration.to_csv('duration.csv')   