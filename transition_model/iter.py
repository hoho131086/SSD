#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 00:40:55 2022

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
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


param_grid = {
    'label_type': ['binary', 'four'],
    'char': ['main_last', 'main_overall']
}
python_path=f'./run_copy_2.py'

dir_path=os.path.dirname(python_path)
python_path=os.path.basename(python_path)
for idx,param in tqdm(enumerate(ParameterGrid(param_grid))):
    #if idx <80: continue
    # print(idx, param)
    arg_now=' '.join(['--{} {} '.format(key,v) for key,v in param.items()])
    exec_line=f'python {python_path} {arg_now}'
    print ('--',idx,'-- ',exec_line)
    subprocess.call(exec_line,shell=True)




