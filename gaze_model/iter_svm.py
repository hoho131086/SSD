#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 00:40:53 2022

@author: shaohao
"""

import os
import argparse
import subprocess 
#import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from glob import glob
import csv
os.environ['MKL_THREADING_LAYER'] = 'GNU'

param_grid = {
    'file':['./results/multi_svm.csv'],
    'feature':['gaze_angle_x,gaze_angle_y',
                'head_angle_x,head_angle_y',
                'gaze_angle_x,head_angle_x',
                # 'gaze_angle_x,gaze_angle_y,head_angle_x,head_angle_y',
                'gaze_angle_x,gaze_angle_y,head_angle_x,head_angle_y,head_angle_z,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_1_z']
                # 'gaze_angle_x,gaze_angle_y,head_angle_x,head_angle_y,head_angle_z,head_pos_x,head_pos_y,head_pos_z,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_1_z']
    # 'features':['gaze_angle_x,gaze_angle_y,head_angle_x,head_angle_y,head_angle_z,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_1_z']
}
python_path=f'./run.py'

with open('./results/multi_svm.csv', 'w') as out:
    writer = csv.writer(out)
    writer.writerow(['features', 'acc', 'f1score', 'recall', 'precision', 'uar'])

dir_path=os.path.dirname(python_path)
python_path=os.path.basename(python_path)
for idx,param in tqdm(enumerate(ParameterGrid(param_grid))):
    # if idx <3: continue
    # print(idx, param)
    arg_now=' '.join(['--{} {} '.format(key,v) for key,v in param.items()])
    exec_line=f'python {python_path} {arg_now}'
    print ('--',idx,'-- ',exec_line)
    subprocess.call(exec_line,shell=True)
