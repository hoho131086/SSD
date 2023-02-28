#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:23:29 2022

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

def trans_label_convert(bef, now):
    if bef == 0 and now == 0:
        return 0
    
    elif bef == 0 and now == 1:
        return 1
    
    elif bef == 1 and now == 0:
        return 2
    
    elif bef == 1 and now == 1:
        return 3
    
    else:pdb.set_trace()

def get_talk_data(file_path):
    if file_path != 'start':
        talk_file = joblib.load(file_path)
    
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
    else:
        talk_file_stretch = np.zeros((300,))
        
    return talk_file_stretch

def get_gaze_data(gaze_mode, file_path):    
    gaze_data = []
    if file_path != 'start':
        member = int(os.path.basename(file_path).split('.')[0][-1])
        other_member = get_member_id(member)
        all_member = [member]+other_member
        
        row_idx= os.path.basename(file_path).split('_')[1]
        
        if gaze_mode == 'gaze_model':
            gaze_dir = os.path.dirname(file_path)
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
            
        elif gaze_mode == 'openface':
            for mem in all_member:
                gaze_dir = os.path.dirname(file_path)
                gaze_dir = gaze_dir.replace('asd_multi_result', 'openface_clip')
                
                gaze_filename = 'row_{}_scores_{}.pckl'.format(row_idx, mem)
                other_gaze = joblib.load(os.path.join(gaze_dir, gaze_filename))
                assert len(other_gaze)==300, pdb.set_trace()
                
                gaze_data.append(other_gaze)
        
        elif gaze_mode == 'no':
            gaze_data = [np.array(i) for i in range(4)]
    
    else:
        gaze_data.append(np.zeros((300,4)))
        gaze_data.append(np.zeros((300,1)))
        gaze_data.append(np.zeros((300,1)))
        gaze_data.append(np.zeros((300,1)))
    
    return gaze_data

def get_other_talk(path, current):
    
    other = get_member_id(current)
    
    output = []
    for oth in other:
        temp_path = path.replace('_scores_{}'.format(current), '_scores_{}'.format(oth))
        oth_talk = get_talk_data(temp_path)
        oth_talk[oth_talk>0]=1
        oth_talk[oth_talk<0]=0
        oth_talk = 1 - oth_talk
        output.append(oth_talk)
    
    return output

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def get_other_gaze_with_mask(mask, gaze, mode):
    
    if mode == 'all':
        output = [(mask[i], gaze[i+1]) for i in range(3)]
    
    elif mode == 'main_overall':
        chose_member = np.argmin([sum(mask[i]) for i in range(3)])
        output = [(mask[chose_member], gaze[chose_member+1])]
    
    elif mode == 'main_last':
        fine = []
        for i in range(3):
            if len(np.where(mask[i]==0)[0]) == 0:
                continue
            find_consecute = consecutive(np.where(mask[i]==0)[0])
            fine.append([i,find_consecute[-1][-1], len(find_consecute[-1]), len(np.where(mask[i]==0)[0])])    
        if len(fine)!=0:
            index = 1
            while len(fine)>1 and index < 4:
                compare_key = max(fine, key=lambda x: x[index])[index]
                fine = [items for items in fine if items[index] == compare_key]
                index += 1 
            
            # pdb.set_trace()  
            output = [(mask[fine[0][0]], gaze[fine[0][0]+1])]
        else:
            output = [(mask[0], gaze[1])]
        
    return output

def get_other_gaze_character(mask, gaze, mode):
    
    if mode == 'main_overall':
        chose_member = np.argmin([sum(mask[i]) for i in range(3)])
        return gaze[chose_member+1]
    
    elif mode == 'main_last':
        fine = []
        for i in range(3):
            if len(np.where(mask[i]==0)[0]) == 0:
                continue
            find_consecute = consecutive(np.where(mask[i]==0)[0])
            fine.append([i+1,find_consecute[-1][-1], len(find_consecute[-1]), len(np.where(mask[i]==0)[0])])    
            
        if len(fine)!=0:
            index = 1
            while len(fine)>1 and index < 4:
                compare_key = max(fine, key=lambda x: x[index])[index]
                fine = [items for items in fine if items[index] == compare_key]
                index += 1 
            
            # pdb.set_trace()  
            output = gaze[fine[0][0]]
        else:
            output = gaze[1]
        # pdb.set_trace()
        return output
       
class next_speaker_Dataset_attention(Dataset):
    def __init__(self, mode, gaze_fea_mode, recording_choose, weighted, label_type, other_gaze_type):
        self.label_type = label_type
        self.other_gaze_type = other_gaze_type
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
        
        bef_label = []
        now_label = []
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
                
                origin_label = list(temp['label_{}'.format(member)]).copy()
                now_label.extend(origin_label)
                origin_len = len(origin_label)
                temp_label = list(temp['label_{}'.format(member)]).copy()
                temp_label.insert(0,0)
                temp_label = temp_label[:origin_len]
                bef_label.extend(temp_label)
                assert len(temp_label) == len(origin_label)
                if self.label_type == 'binary':
                    trans_label = [int(temp_label[i]!=origin_label[i]) for i in range(origin_len)]
                elif self.label_type == 'four':
                    trans_label = [trans_label_convert(temp_label[i],origin_label[i]) for i in range(origin_len)]
                label.extend(trans_label)
                
                origin_path = sorted(glob(talk_recording_path+'/*_{}.pckl'.format(member)))
                path_len = len(origin_path)
                temp_path = origin_path.copy()
                temp_path.insert(0, 'start')
                temp_path = temp_path[:path_len]
                assert len(temp_path) == len(origin_path)
                talk_path.extend([[temp_path[i], origin_path[i]] for i in range(path_len)])
                # pdb.set_trace()
            
        # assert len(label)==(len(sample)*4), pdb.set_trace()
        assert len(label)==len(talk_path), pdb.set_trace()
        assert len(label)==len(recording_lst)
        assert len(label)==len(frame)
        assert len(label)==len(now_label)
        assert len(label)==len(bef_label)
        
        self.data = [[recording_lst[i], label[i], talk_path[i], frame[i], now_label[i], bef_label[i]] for i in range(len(label))]
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
                elif i == 2:
                    self.data_weight.append(self.label_weight[2])
                elif i == 3:
                    self.data_weight.append(self.label_weight[3])
           
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        recording = self.data[idx][0]
        label = self.data[idx][1]
        talk_file_path_before = self.data[idx][2][0]
        talk_file_path_now = self.data[idx][2][1]
        # pdb.set_trace()
        now_label = self.data[idx][-2]
        bef_label = self.data[idx][-1]
        current_member = int(talk_file_path_now.split('_scores_')[-1][0])
        
        start_frame = int(np.round(self.data[idx][3]*30, 0))+1
        
        talk_data_before = get_talk_data(talk_file_path_before)
        talk_data_now = get_talk_data(talk_file_path_now)
        
        gaze_data_before = get_gaze_data(gaze_mode = self.gaze_fea_mode, file_path = talk_file_path_before)
        gaze_data_now = get_gaze_data(gaze_mode = self.gaze_fea_mode, file_path = talk_file_path_now)
        
        other_mask_before = get_other_talk(talk_file_path_before, current_member)
        other_mask_now = get_other_talk(talk_file_path_now, current_member)
        # pdb.set_trace()
        
        char_gaze_before = get_other_gaze_character(other_mask_before, gaze_data_before, self.other_gaze_type)
        char_gaze_now = get_other_gaze_character(other_mask_now, gaze_data_now,  self.other_gaze_type)
        
        before_data = (talk_data_before, gaze_data_before, char_gaze_before)
        now_data = (talk_data_now, gaze_data_now, char_gaze_now)
        # pdb.set_trace()
        
        return before_data, now_data, label, talk_file_path_now

        # look for main other speaker (two cahracter)



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
    a = next_speaker_Dataset_attention('all', 'gaze_model', recording_choose = recording_all, weighted = 'no', label_type='binary', other_gaze_type = 'main_last')    
    
    loader = DataLoader(dataset = a, batch_size = 64, shuffle=False)
    for step, alls in tqdm.tqdm(enumerate(loader)):
        # print(batch_x.shape)
        pdb.set_trace()
        C=0

    stop = timeit.default_timer()
    print(stop-start)










    