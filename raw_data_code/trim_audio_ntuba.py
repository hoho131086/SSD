#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 04:31:09 2022

@author: shaohao
"""

import audiofile
import opensmile
from glob import glob
import pandas as pd
import os
import tqdm


data_path = '/homes/GPU2/shaohao/Corpus/NTUBA/multimediate_forum/*'
ori_path = '/homes/GPU2/shaohao/Corpus/NTUBA/audio_from_video/'
output_dir = '/homes/GPU2/shaohao/Corpus/NTUBA/trim/'
sox_path = './output.sh'

if os.path.isfile(sox_path):
    os.remove(sox_path)

#%%

key = 0
for data in tqdm.tqdm(sorted(glob(data_path))):
    # print(data)
    df = pd.read_csv(data)
    current_recording = os.path.basename(data).split('.')[0]
    audio_path = ori_path+current_recording+'/{}.wav'.format(current_recording)
    
    for ind, row in df.iterrows():
        start_time = row['Start_desktop']
        
        if not os.path.isdir(output_dir+current_recording):
            os.mkdir(output_dir+current_recording)
        
        output_file_name = output_dir+current_recording+'/'+'{}.wav'.format(str(ind).zfill(6))
        
        # print(output_file_name)
    
        with open(sox_path, 'a') as f:
            f.write('sox ')
            f.write(audio_path+' '+ output_file_name+ ' trim ' + str(start_time) + ' 10\n')
    
    
    