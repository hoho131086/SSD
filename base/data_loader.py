#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:05:10 2022

@author: shaohao
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence 

import pandas as pd
from glob import glob
import numpy as np
from collections import Counter
import scipy.interpolate as interp

import joblib
import random
import os
import sys
import tqdm
import timeit

import pdb

def get_member_id(candi):
    output = [candi+1, candi+2, candi+3]
    output = [i-4 if i > 4 else i for i in output]
        
    return output

def get_dir(current_member, others):
    
    if (others - current_member == 1) or (others - current_member == -3):
        return 'right'
    elif (others - current_member == 2) or (others - current_member == -2):
        return 'middle'
    elif (others - current_member == -1) or (others - current_member == 3):
        return 'left'
    else:
        pdb.set_trace()
        
class next_speaker_Dataset_attention(Dataset):
    def __init__(self, mode, gaze_fea_mode, recording_choose, weighted):
        self.mode = mode
        self.recording_choose = recording_choose
        if mode == 'train':
            sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
        elif mode == 'test':
            sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
            self.recording_choose = sorted(set(sample['recording']))
        elif mode == 'all':
            sample_train = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
            sample_val = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
            sample = pd.concat([sample_train, sample_val]).reset_index(drop=True)
            self.recording_choose = sorted(set(sample['recording']))
            # pdb.set_trace()
        
        talk_root = '/homes/GPU2/shaohao/Corpus/multimediate/asd_multi_result/'
        
        label = []
        talk_path = []
        recording_lst = []
        frame = []
        self.data = []
        for recording in self.recording_choose:
            temp = sample.copy()
            temp = temp[temp['recording']==recording]
            talk_recording_path = talk_root+recording
            recording_len = len(temp)
            for member in range(1,5):
                recording_lst.extend([recording for i in range(recording_len)])
                frame.extend(list(temp['start_time']))
                label.extend(temp['label_{}'.format(member)])
                talk_path.extend(sorted(glob(talk_recording_path+'/*_{}.pckl'.format(member))))
            # pdb.set_trace()
            
        # assert len(label)==(len(sample)*4), pdb.set_trace()
        assert len(label)==len(talk_path), pdb.set_trace()
        assert len(label)==len(recording_lst)
        assert len(label)==len(frame)
        
        self.data = [[recording_lst[i], label[i], talk_path[i], frame[i]] for i in range(len(label))]
        self.gaze_fea_mode = gaze_fea_mode
        # pdb.set_trace()
        
        if weighted == 'yes':
            self.data_weight = []
            # self.label_weight = [1, 1]
            self.label_weight = [1/item[1] for item in sorted(Counter(label).items())]
            for i in label:
                if i == 0:
                    self.data_weight.append(self.label_weight[0])
                elif i == 1:
                    self.data_weight.append(self.label_weight[1])
           
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        recording = self.data[idx][0]
        label = self.data[idx][1]
        talk_file_path = self.data[idx][2]
        row_idx = os.path.basename(talk_file_path).split('_')[1]
        
        start_frame = int(np.round(self.data[idx][3]*30, 0))+1
        
        member = int(os.path.basename(talk_file_path).split('.')[0][-1])
        other_member = get_member_id(member)
        all_member = [member]+other_member  ## [me, right, middle, left]

        talk_file = joblib.load(talk_file_path)
  
        if len(talk_file)!= 0:
            asd_ind = max((len(l), i) for i, l in enumerate(talk_file))[1]
        else:
            asd_ind = 0
            talk_file = [[-3 for i in range(249)]]
        if len(talk_file[asd_ind])!=249:
            talk_file = np.hstack([talk_file[asd_ind],np.array([-3 for i in range(249)])])
            talk_file = talk_file[:249]
        else:
            talk_file = talk_file[asd_ind]
            
        talk_file = np.array(talk_file)
        interp_talk = interp.interp1d(np.arange(talk_file.size),talk_file)
        talk_file_stretch = interp_talk(np.linspace(0,talk_file.size-1,300))
        # pdb.set_trace()
        
        gaze_data = []
        if self.gaze_fea_mode == 'gaze_model':
            gaze_dir = os.path.dirname(talk_file_path)
            gaze_dir = gaze_dir.replace('asd_multi_result', 'gaze_model_faster')
            gaze_filename = 'row_{}_scores_{}.pckl'.format(row_idx, member)
            gaze_dic = joblib.load(os.path.join(gaze_dir, gaze_filename))
            
            for key in ['me', 'right', 'middle', 'left']:
                if key == 'me':
                    gaze_data.append(gaze_dic[key].values.astype(float))
                else:
                    temp = gaze_dic[key]
                    temp = temp.iloc[:, 1].values.astype(float)
                    gaze_data.append(temp.reshape(-1,1))
            
            # pdb.set_trace()
                
        elif self.gaze_fea_mode == 'openface':
            for mem in all_member:
                gaze_dir = os.path.dirname(talk_file_path)
                gaze_dir = gaze_dir.replace('asd_multi_result', 'openface_clip')
                
                gaze_filename = 'row_{}_scores_{}.pckl'.format(row_idx, mem)
                other_gaze = joblib.load(os.path.join(gaze_dir, gaze_filename))
                assert len(other_gaze)==300, pdb.set_trace()
                
                gaze_data.append(other_gaze)
        
        elif self.gaze_fea_mode == 'no':
            gaze_data = [np.array(i) for i in range(4)]
        
        return talk_file_stretch, gaze_data, label, talk_file_path





#%%


if __name__=='__main__':  

    recording_all = [
                     'recording07',
                     'recording08',
                     'recording09',
                     'recording10',
                     'recording11',
                     'recording12',
                     'recording13',
                     'recording14',
                     'recording15',
                     'recording16',
                     'recording17',
                     'recording18',
                     'recording19',
                     'recording20',
                     'recording21',
                     'recording22',
                     'recording23',
                     'recording24',
                     'recording25',
                     'recording26',
                     'recording27',
                     'recording28'
                     ]
    start = timeit.default_timer()
    # a = next_speaker_Dataset_talk('train', recording_choose = ['recording07'])
    a = next_speaker_Dataset_attention('all', 'gaze_model', recording_choose = recording_all, weighted = 'no')    
    
    loader = DataLoader(dataset = a, batch_size = 64, shuffle=True)
    for step, (talk, gaze, label, filename) in tqdm.tqdm(enumerate(loader)):
        # print(batch_x.shape)
        # pdb.set_trace()
        C=0

    stop = timeit.default_timer()
    print(stop-start)










    