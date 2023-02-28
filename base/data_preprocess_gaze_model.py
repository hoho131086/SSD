#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 23:47:39 2022

@author: shaohao
"""
import pandas as pd
import numpy as np

import os
import joblib
from glob import glob
import tqdm
import pdb

def turn_frame(df):
    df['start_time'] = int(np.round(df['start_time']/(1/30)))+1
    df['end_time'] = int(np.round(df['end_time']/(1/30)))+1
    return df

def get_other(candi):
    output = [candi+i for i in range(1,4)]
    output = [i-4 if i > 4 else i for i in output]
    return output


sample_df = pd.DataFrame()
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
sample_df = sample_df.reset_index(drop=True)
sample_df = sample_df.apply(turn_frame, axis=1)
recording_lst = sorted(set(sample_df['recording']))

#%%

OUTPUT_DIR = '/homes/GPU2/shaohao/Corpus/multimediate/gaze_model_faster/'

for recording in tqdm.tqdm(recording_lst):
    
    temp_sample = sample_df.copy()
    temp_sample = temp_sample[temp_sample['recording']==recording].reset_index(drop=True)
    
    gaze_dir = '/homes/GPU2/shaohao/Corpus/multimediate/gaze_model_inference/{}/'.format(recording)
    output_dir = '/homes/GPU2/shaohao/Corpus/multimediate/gaze_model_faster/{}'.format(recording)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    
    for member in range(1,5):
        
        me_gaze = pd.read_csv(gaze_dir+'{}_inference.csv'.format(member), index_col=0)
        
        other = get_other(member)
        right_gaze =  pd.read_csv(gaze_dir+'{}_inference.csv'.format(other[0]))
        right_gaze = right_gaze[['frame', 'look_empty_prob', 'look_left_prob']]
        
        middle_gaze =  pd.read_csv(gaze_dir+'{}_inference.csv'.format(other[1]))
        middle_gaze = middle_gaze[['frame', 'look_empty_prob', 'look_middle_prob']]
        
        left_gaze =  pd.read_csv(gaze_dir+'{}_inference.csv'.format(other[2]))
        left_gaze = left_gaze[['frame', 'look_empty_prob', 'look_right_prob']]
        # pdb.set_trace()
        for ind, row in temp_sample.iterrows():
            member_output = {}
            start_time = row['start_time']
            
            temp_me = me_gaze.copy()
            temp_me = temp_me.iloc[start_time:start_time+300, :]
            temp_me = temp_me.drop(columns = ['frame', 'recording'])
            temp_me = temp_me.astype(float)
            if len(temp_me)!=300:    
                pdb.set_trace()
            
            temp_right = right_gaze.copy()
            temp_right = temp_right.iloc[start_time:start_time+300, :]
            temp_right = temp_right.drop(columns = ['frame'])
            temp_right = temp_right.astype(float)
            if len(temp_right)!=300:    
                pdb.set_trace()
                
            temp_middle = middle_gaze.copy()
            temp_middle = temp_middle.iloc[start_time:start_time+300, :]
            temp_middle = temp_middle.drop(columns = ['frame'])
            temp_middle = temp_middle.astype(float)
            if len(temp_middle)!=300:    
                pdb.set_trace()
            
            temp_left = left_gaze.copy()
            temp_left = temp_left.iloc[start_time:start_time+300, :]
            temp_left = temp_left.drop(columns = ['frame'])
            temp_left = temp_left.astype(float)
            if len(temp_left)!=300:
                pdb.set_trace()
            
            member_output['me'] = temp_me
            member_output['right'] = temp_right
            member_output['middle'] = temp_middle
            member_output['left'] = temp_left

            joblib.dump(member_output, os.path.join(output_dir, 'row_{}_scores_{}.pckl'.format(str(ind).zfill(3), member)))
    
    
    
#%% check

for recording in tqdm.tqdm(recording_lst):
    
    temp_sample = sample_df.copy()
    temp_sample = temp_sample[temp_sample['recording']==recording].reset_index(drop=True)
    output_dir = '/homes/GPU2/shaohao/Corpus/multimediate/openface_clip/{}'.format(recording)
    
    if len(glob(output_dir+'/*')) != len(temp_sample)*4:
        print(recording)
    
    
    
    
    
    
    
    
    
    
    
    



