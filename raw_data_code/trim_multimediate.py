#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 23:26:22 2021

@author: shaohao
"""


from glob import glob
import pandas as pd
import os


data_path = ["/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv",
             "/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv"]
ori_path = '/homes/GPU2/shaohao/Corpus/multimediate/data/'
output_dir = '/homes/GPU2/shaohao/Corpus/multimediate/trim/'
sox_path = './output.sh'

if os.path.isfile(sox_path):
    os.remove(sox_path)

#%%
key = 0
for data in data_path:
    df = pd.read_csv(data)
    for ind, row in df.iterrows():
        current_recording = row['recording']
        audio_path = ori_path+current_recording+'/audio.wav'
        start_time = row['start_time']
        
        if not os.path.isdir(output_dir+current_recording):
            os.mkdir(output_dir+current_recording)
        
        output_file_name = output_dir+current_recording+'/'+'{}.wav'.format(str(ind).zfill(6))
        
        # print(output_file_name)
    
        with open(sox_path, 'a') as f:
            f.write('sox ')
            f.write(audio_path+' '+ output_file_name+ ' trim ' + str(start_time) + ' 10\n')
    
    
    