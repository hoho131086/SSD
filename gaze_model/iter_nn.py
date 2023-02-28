#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:03:54 2022

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

# ['recording', 'member','label', 
# 'gaze_angle_x', 'gaze_angle_y', 
# 'head_angle_x', 'head_angle_y', 'head_angle_z',
# 'head_pos_x', 'head_pos_y', 'head_pos_z', 
# 'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
# 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']



param_grid = {
    'output_path':['./results/multi_nn_all_group_try_var_nopose.csv'],
    'epochs': [30, 60, 90],
    'LR':[1e-3, 1e-2, 5e-2],
    'dropout':[0, 0.3],
    'layer_num': [2,3,4],
    'node_num': [8,16,32,64,128],
    'batch_size': [256],
    'output_dim':[4],
    'target_group': ['all'],
    'std':['yes', 'no'],
    # 'features':['gaze_angle_x,gaze_angle_x',
    #             'head_angle_x,head_angle_y',
    #             'gaze_angle_x,head_angle_x',
    'features':['gaze_angle_x,gaze_angle_y,head_angle_x,head_angle_y,head_angle_z,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_1_z']
}
python_path=f'./run_nn.py'

with open('./results/multi_nn_all_group_try_var_nopose.csv', 'w') as out:
    writer = csv.writer(out)
    writer.writerow(['cv', 'std','learning_rate', 'batch_size', 'layer_num', 'node_num', 'dropout', 'epoch', 'features', 'acc', 'f1score', 'recall', 'precision', 'uar'])

dir_path=os.path.dirname(python_path)
python_path=os.path.basename(python_path)
for idx,param in tqdm(enumerate(ParameterGrid(param_grid))):
    # if idx <3: continue
    # print(idx, param)
    arg_now=' '.join(['--{} {} '.format(key,v) for key,v in param.items()])
    exec_line=f'python {python_path} {arg_now}'
    print ('--',idx,'-- ',exec_line)
    subprocess.call(exec_line,shell=True)
