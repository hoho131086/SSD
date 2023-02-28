#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:49:56 2022

@author: shaohao
"""

from glob import glob
import pandas as pd
import os
import subprocess 
import tqdm

output_dir = '/homes/GPU2/shaohao/Corpus/NTUBA/video_trim_session/'

for files in tqdm.tqdm(sorted(glob('/homes/GPU2/shaohao/Corpus/NTUBA/multimediate_forum/*.csv'))):
    df = pd.read_csv(files)
    group = os.path.basename(files).split('.')[0].zfill(2)
    # print(group)
    if not os.path.isdir(output_dir+group):
        os.mkdir(output_dir+group)

    temp = df.copy()
    task_1 = temp[temp['Tag']=='Task1']
    task_1_time = [task_1.loc[task_1.index[0], 'Start_desktop'], task_1.loc[task_1.index[-1], 'End_desktop']]
    
    temp = df.copy()
    task_2 = temp[temp['Tag']=='Task2']
    task_2_time = [task_2.loc[task_2.index[0], 'Start_desktop'], task_2.loc[task_2.index[-1], 'End_desktop']]
    
    temp = df.copy()
    reflect = temp[temp['Tag']=='Reflection']
    reflect_time = [reflect.loc[reflect.index[0], 'Start_desktop'], reflect.loc[reflect.index[-1], 'End_desktop']]
    
    video_lst = sorted(glob('/homes/GPU2/shaohao/Corpus/NTUBA/Non_group/{}*.wmv'.format(group)))
    for video in video_lst:
        member_id = os.path.basename(video).split('.')[0]
        output_video = video.replace('Non_group', 'video_trim_session/{}'.format(group))
        temp_name = output_video.replace('.wmv', '_task1.wmv')
        exec_line = ' ffmpeg -ss {} -to {}  -i {} -c copy {}'.format(task_1_time[0], task_1_time[1], video, temp_name)
        subprocess.call(exec_line,shell=True)
        
        temp_name = output_video.replace('.wmv', '_task2.wmv')
        exec_line = ' ffmpeg -ss {} -to {}  -i {} -c copy {}'.format(task_2_time[0], task_2_time[1], video, temp_name)
        subprocess.call(exec_line,shell=True)
        
        temp_name = output_video.replace('.wmv', '_reflect.wmv')
        exec_line = ' ffmpeg -ss {} -to {}  -i {} -c copy {}'.format(reflect_time[0], reflect_time[1], video, temp_name)
        subprocess.call(exec_line,shell=True)
    






