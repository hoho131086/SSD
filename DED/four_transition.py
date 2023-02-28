#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 23:16:48 2022

@author: shaohao
"""

import joblib

mode = 'test'
method = 'indiv'
fea = 'indiv/cv4'

trans = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/final/DED/idx/transit.pkl")
length = joblib.load('data/{}/{}_{}_dialog.pkl'.format(fea,method, mode))


for key in length:
    total_length = len(length[key])    
    trans_ = [1 if i in trans[key] else 0 for i in range(total_length)]
    



