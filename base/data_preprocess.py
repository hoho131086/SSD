#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 21:54:27 2022

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


sample_df = pd.DataFrame()
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
sample_df = sample_df.reset_index(drop=True)
sample_df = sample_df.apply(turn_frame, axis=1)
recording_lst = sorted(set(sample_df['recording']))
gaze_fea = ['frame', ' pose_Rx', ' pose_Ry', ' pose_Rz', ' gaze_angle_x', ' gaze_angle_y',
            ' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z']


OUTPUT_DIR = '/homes/GPU2/shaohao/Corpus/multimediate/openface_clip/'

for recording in tqdm.tqdm(recording_lst):
    
    temp_sample = sample_df.copy()
    temp_sample = temp_sample[temp_sample['recording']==recording].reset_index(drop=True)
    
    gaze_dir = '/homes/GPU2/shaohao/Corpus/multimediate/openface_result/{}/'.format(recording)
    output_dir = '/homes/GPU2/shaohao/Corpus/multimediate/openface_clip/{}'.format(recording)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for member in range(1,5):
        other_gaze = pd.read_csv(gaze_dir+'subjectPos{}.video.csv'.format(member))
        other_gaze = other_gaze[gaze_fea]
        for ind, row in temp_sample.iterrows():
            temp_gaze = other_gaze.copy()
            start_time = row['start_time']
            temp_gaze = temp_gaze.iloc[start_time:start_time+300, :]
            temp_gaze = temp_gaze.drop(columns = ['frame'])
            temp_gaze = temp_gaze.values.astype(float)
            # pdb.set_trace()
            if len(temp_gaze)!=300:    
                pdb.set_trace()
            joblib.dump(temp_gaze, os.path.join(output_dir, 'row_{}_scores_{}.pckl'.format(str(ind).zfill(3), member)))
    
    
    
#%% check

for recording in tqdm.tqdm(recording_lst):
    
    temp_sample = sample_df.copy()
    temp_sample = temp_sample[temp_sample['recording']==recording].reset_index(drop=True)
    output_dir = '/homes/GPU2/shaohao/Corpus/multimediate/openface_clip/{}'.format(recording)
    
    if len(glob(output_dir+'/*')) != len(temp_sample)*4:
        print(recording)
    
    
    
    
    
    
    
    
    
    
    
    



