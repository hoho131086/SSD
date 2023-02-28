#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 00:34:24 2021

@author: shaohao
"""
from glob import glob
import pandas as pd
import numpy as np
import os
import opensmile
import audiofile
import tqdm
import joblib

data_path = ["/homes/GPU2/shaohao/multimediate/sample_lists/train_next_speaker.csv",
             "/homes/GPU2/shaohao/multimediate/sample_lists/val_next_speaker.csv"]
# audio_lst = sorted(glob('/homes/GPU2/shaohao/multimediate/trim/*/*'))


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

#%%
all_df = pd.DataFrame()
data = "/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv"
df = pd.read_csv(data)

for ind, row in tqdm.tqdm(df.iterrows()):
    # print("Processing file {} {}".format(row['recording'], ind))
    current_record = row['recording']
    index = row['index']
    find_path = '/homes/GPU2/shaohao/Corpus/multimediate/trim/{}/{}.wav'.format(current_record, str(index).zfill(6))
    if not os.path.isfile(find_path):
        print(find_path)
    
    duration = audiofile.duration(find_path)
    if duration < 10:
        import pdb;pdb.set_trace()
        print(find_path)
    signal, sampling_rate = audiofile.read(find_path,always_2d=True)
    egemaps = smile.process_signal(signal,sampling_rate)
    egemaps = egemaps.reset_index(drop=True)
    # import pdb;pdb.set_trace()
    temp = pd.concat([row, egemaps.loc[0]])
    all_df = pd.concat([all_df, temp], axis=1)
    # import pdb;pdb.set_trace()

all_df = all_df.T.reset_index(drop=True)
joblib.dump(all_df,'/homes/GPU2/shaohao/turn_taking/turn_changing/Data/multimediate_egemap_train.pkl')

#%%
all_df = pd.DataFrame()
data = "/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv"
df = pd.read_csv(data)

for ind, row in tqdm.tqdm(df.iterrows()):
    current_record = row['recording']
    index = row['index']
    find_path = '/homes/GPU2/shaohao/Corpus/multimediate/trim/{}/{}.wav'.format(current_record, str(index).zfill(6))
    if not os.path.isfile(find_path):
        print(find_path)
    
    duration = audiofile.duration(find_path)
    if duration < 10:
        import pdb;pdb.set_trace()
        print(find_path)
    signal, sampling_rate = audiofile.read(find_path,always_2d=True)
    egemaps = smile.process_signal(signal,sampling_rate)
    egemaps = egemaps.reset_index(drop=True)
    temp = pd.concat([row, egemaps.loc[0]])
    all_df = pd.concat([all_df, temp], axis=1)
    # import pdb;pdb.set_trace()

all_df = all_df.T.reset_index(drop=True)
joblib.dump(all_df,'/homes/GPU2/shaohao/turn_taking/turn_changing/Data/multimediate_egemap_valid.pkl')


