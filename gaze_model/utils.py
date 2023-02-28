#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:31:30 2022

@author: shaohao
"""

from torch.utils.data import Dataset
import joblib
from sklearn import preprocessing
import pandas as pd
import numpy as np 
import os
import pdb
from sklearn.preprocessing import StandardScaler

def get_direction(look_at, member):
    if float(look_at) == 0: 
        return 0
    # pdb.set_trace()
    direction =float(look_at)-float(member)
    if direction<0:
        direction+=4
    
    return direction

def get_op(recording, member):
    op = pd.read_csv('/homes/GPU2/shaohao/Corpus/multimediate/openface_result/{}/subjectPos{}.video.csv'.format(recording, member))
    op = op[['frame', ' pose_Tx', ' pose_Ty', ' pose_Tz', ' pose_Rx', ' pose_Ry', ' pose_Rz', ' gaze_angle_x', ' gaze_angle_y',
             ' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z']]
    op = op.rename(columns={' gaze_angle_x': '{}_gaze_angle_x'.format(member), ' gaze_angle_y': '{}_gaze_angle_y'.format(member), 
                            ' pose_Rx': '{}_head_x'.format(member), ' pose_Ry': '{}_head_y'.format(member), ' pose_Rz': '{}_head_z'.format(member),
                            ' pose_Tx': '{}_pose_x'.format(member), ' pose_Ty': '{}_pose_y'.format(member), ' pose_Tz': '{}_pose_z'.format(member), 
                            ' gaze_0_x': '{}_gaze_0_x'.format(member), ' gaze_0_y': '{}_gaze_0_y'.format(member), ' gaze_0_z': '{}_gaze_0_z'.format(member),
                            ' gaze_1_x': '{}_gaze_1_x'.format(member), ' gaze_1_y': '{}_gaze_1_y'.format(member), ' gaze_1_z': '{}_gaze_1_z'.format(member)})
    
    return op

def get_anno_multi():
    anno = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/labels/eye_contact_annotation.csv")
    anno = anno.rename(columns={'Unnamed: 0': 'frame'})
    recording_set = [i.split('.')[0] for i in anno.columns]
    recording_set.remove('frame')
    recording_set = sorted(list(set(recording_set)))
    anno_clean = pd.DataFrame()
    
    training_set = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_eye_contact.csv")
    training_set = sorted(set(training_set['recording']))
    
    testing_set = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_eye_contact.csv")
    testing_set = sorted(set(testing_set['recording']))
    
    
    for recording in training_set:
        temp_anno = anno.copy()
        temp_col = [col for col in temp_anno.columns if recording in col]
        temp_anno = temp_anno[['frame']+temp_col]
        temp_anno = temp_anno.rename(columns=temp_anno.iloc[0])
        temp_anno = temp_anno.rename(columns={temp_anno.columns[0]: 'frame'})
        temp_anno = temp_anno.drop(index=0)
        sub_col = temp_anno.columns[-4:]
        # pdb.set_trace()
        temp_anno = temp_anno.dropna(axis = 0, subset = sub_col, how = 'all')
        lack_member = temp_anno.columns[temp_anno.isna().all()].tolist()
        exist_member = temp_anno.columns[~temp_anno.isna().all()].tolist()
        # if len(lack_member) > 0:
        #     continue
        temp_anno = temp_anno.fillna(0)
        temp_anno['frame'] += 1
        
        # for member in sub_col:
        #     if member in exist_member:
        #         temp_anno[member] = temp_anno[member].apply(get_direction, member=int(member.split('Pos')[1]))
        #     else:
        #         # pdb.set_trace()
        #         temp_anno[member] = [4 for i in range(len(temp_anno))]
        for member in sub_col:
            temp_anno['{}_dir'.format(member)] = temp_anno[member].apply(get_direction, member=int(member.split('Pos')[-1]))
        
        
       
        for member in range(1,5):
            op = get_op(recording, member)     
            temp_anno = temp_anno.merge(op, on='frame')
        
        # pdb.set_trace()
        temp_anno = temp_anno.astype(float)
        temp_anno['recording'] = recording
        anno_clean = pd.concat([anno_clean, temp_anno], axis=0)
        
    # pdb.set_trace()
    joblib.dump(anno_clean, '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/all_group/four_class/multi_gaze_training_all_group.pkl')

    anno_clean = pd.DataFrame()
    for recording in testing_set:
        temp_anno = anno.copy()
        temp_col = [col for col in temp_anno.columns if recording in col]
        temp_anno = temp_anno[['frame']+temp_col]
        temp_anno = temp_anno.rename(columns=temp_anno.iloc[0])
        temp_anno = temp_anno.rename(columns={temp_anno.columns[0]: 'frame'})
        temp_anno = temp_anno.drop(index=0)
        sub_col = temp_anno.columns[-4:]
        # pdb.set_trace()
        temp_anno = temp_anno.dropna(axis = 0, subset = sub_col, how = 'all')
        lack_member = temp_anno.columns[temp_anno.isna().all()].tolist()
        exist_member = temp_anno.columns[~temp_anno.isna().all()].tolist()
        # if len(lack_member) > 0:
        #     continue
        temp_anno = temp_anno.fillna(0)
        temp_anno['frame'] = temp_anno['frame']+1

        # for member in sub_col:
        #     if member in exist_member:
        #         temp_anno[member] = temp_anno[member].apply(get_direction, member=int(member.split('Pos')[1]))
        #     else:
        #         # pdb.set_trace()
        #         temp_anno[member] = [4 for i in range(len(temp_anno))]
        for member in sub_col:
            temp_anno['{}_dir'.format(member)] = temp_anno[member].apply(get_direction, member=int(member.split('Pos')[1]))
        
        # pdb.set_trace()
        
        for member in range(1,5):
            op = get_op(recording, member)     
            temp_anno = temp_anno.merge(op, on='frame')
        
        # pdb.set_trace()
        temp_anno = temp_anno.astype(float)
        temp_anno['recording'] = recording
        anno_clean = pd.concat([anno_clean, temp_anno], axis=0)
        
    
    joblib.dump(anno_clean, '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/all_group/four_class/multi_gaze_testing_all_group.pkl')

def stretch_multi_op():
    
    scaler = StandardScaler()
    df = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/all_group/four_class/multi_gaze_training_all_group.pkl")
    output = pd.DataFrame(columns=['recording', 'frame', 'member','dir_label', 'id_label', 
                                   'gaze_angle_x', 'gaze_angle_y', 
                                   'head_angle_x', 'head_angle_y', 'head_angle_z',
                                   'head_pos_x', 'head_pos_y', 'head_pos_z', 
                                   'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                                   'gaze_1_x', 'gaze_1_y', 'gaze_1_z'])
    
    for ind, row in df.iterrows():
        for member in range(1,5):
            temp = pd.DataFrame(columns=['recording', 'frame', 'member','dir_label', 'id_label',
                                         'gaze_angle_x', 'gaze_angle_y', 
                                         'head_angle_x', 'head_angle_y', 'head_angle_z',
                                         'head_pos_x', 'head_pos_y', 'head_pos_z', 
                                         'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                                         'gaze_1_x', 'gaze_1_y', 'gaze_1_z'])
            
            temp.loc[0, 'recording'] = row['recording']
            temp.loc[0, 'member'] = member
            temp.loc[0, 'frame'] = row['frame']
            temp.loc[0, 'dir_label'] = row['subjectPos{}_dir'.format(member)]
            temp.loc[0, 'id_label'] = row['subjectPos{}'.format(member)]
            temp.loc[0, 'gaze_angle_x'] = row['{}_gaze_angle_x'.format(member)]
            temp.loc[0, 'gaze_angle_y'] = row['{}_gaze_angle_y'.format(member)]
            temp.loc[0, 'head_angle_x'] = row['{}_head_x'.format(member)]
            temp.loc[0, 'head_angle_y'] = row['{}_head_y'.format(member)]
            temp.loc[0, 'head_angle_z'] = row['{}_head_z'.format(member)]
            temp.loc[0, 'head_pos_x'] = row['{}_pose_x'.format(member)]
            temp.loc[0, 'head_pos_y'] = row['{}_pose_y'.format(member)]
            temp.loc[0, 'head_pos_z'] = row['{}_pose_z'.format(member)]
            temp.loc[0, 'gaze_0_x'] = row['{}_gaze_0_x'.format(member)]
            temp.loc[0, 'gaze_0_y'] = row['{}_gaze_0_y'.format(member)]
            temp.loc[0, 'gaze_0_z'] = row['{}_gaze_0_z'.format(member)]
            temp.loc[0, 'gaze_1_x'] = row['{}_gaze_1_x'.format(member)]
            temp.loc[0, 'gaze_1_y'] = row['{}_gaze_1_y'.format(member)]
            temp.loc[0, 'gaze_1_z'] = row['{}_gaze_1_z'.format(member)]
            
            # pdb.set_trace()
            output = pd.concat([output, temp], axis=0)
    
    output = output.reset_index(drop=True)
    col = ['gaze_angle_x', 'gaze_angle_y', 
           'head_angle_x', 'head_angle_y', 'head_angle_z',
           # 'head_pos_x', 'head_pos_y', 'head_pos_z', 
           'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
           'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
    # scaler.fit(output[col])
    # joblib.dump(scaler, '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/standardscaler.pkl')
    # output[col] = scaler.transform(output[col])
    joblib.dump(output, '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/all_group/four_class/multi_gaze_training_stretch_all_group_.pkl')
   
    df = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/all_group/four_class/multi_gaze_testing_all_group.pkl")
    output = pd.DataFrame(columns=['recording', 'frame', 'member','dir_label', 'id_label', 
                                   'gaze_angle_x', 'gaze_angle_y', 
                                   'head_angle_x', 'head_angle_y', 'head_angle_z',
                                   'head_pos_x', 'head_pos_y', 'head_pos_z', 
                                   'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                                   'gaze_1_x', 'gaze_1_y', 'gaze_1_z'])
    
    for ind, row in df.iterrows():
        for member in range(1,5):
            temp = pd.DataFrame(columns=['recording', 'frame', 'member','dir_label', 'id_label',
                                         'gaze_angle_x', 'gaze_angle_y', 
                                         'head_angle_x', 'head_angle_y', 'head_angle_z',
                                         'head_pos_x', 'head_pos_y', 'head_pos_z', 
                                         'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                                         'gaze_1_x', 'gaze_1_y', 'gaze_1_z'])
            
            temp.loc[0, 'recording'] = row['recording']
            temp.loc[0, 'member'] = member
            temp.loc[0, 'frame'] = row['frame']
            temp.loc[0, 'dir_label'] = row['subjectPos{}_dir'.format(member)]
            temp.loc[0, 'id_label'] = row['subjectPos{}'.format(member)]
            temp.loc[0, 'gaze_angle_x'] = row['{}_gaze_angle_x'.format(member)]
            temp.loc[0, 'gaze_angle_y'] = row['{}_gaze_angle_y'.format(member)]
            temp.loc[0, 'head_angle_x'] = row['{}_head_x'.format(member)]
            temp.loc[0, 'head_angle_y'] = row['{}_head_y'.format(member)]
            temp.loc[0, 'head_angle_z'] = row['{}_head_z'.format(member)]
            temp.loc[0, 'head_pos_x'] = row['{}_pose_x'.format(member)]
            temp.loc[0, 'head_pos_y'] = row['{}_pose_y'.format(member)]
            temp.loc[0, 'head_pos_z'] = row['{}_pose_z'.format(member)]
            temp.loc[0, 'gaze_0_x'] = row['{}_gaze_0_x'.format(member)]
            temp.loc[0, 'gaze_0_y'] = row['{}_gaze_0_y'.format(member)]
            temp.loc[0, 'gaze_0_z'] = row['{}_gaze_0_z'.format(member)]
            temp.loc[0, 'gaze_1_x'] = row['{}_gaze_1_x'.format(member)]
            temp.loc[0, 'gaze_1_y'] = row['{}_gaze_1_y'.format(member)]
            temp.loc[0, 'gaze_1_z'] = row['{}_gaze_1_z'.format(member)]
            
            # pdb.set_trace()
            output = pd.concat([output, temp], axis=0)
    
    output = output.reset_index(drop=True)
    col = ['gaze_angle_x', 'gaze_angle_y', 
           'head_angle_x', 'head_angle_y', 'head_angle_z',
           # 'head_pos_x', 'head_pos_y', 'head_pos_z', 
           'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
           'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
    # output[col] = scaler.transform(output[col])
    joblib.dump(output, '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/all_group/four_class/multi_gaze_testing_stretch_all_group_.pkl')



    
get_anno_multi()
# pdb.set_trace()
stretch_multi_op()
# get_anno_multi()

#%%
def get_anno_multi_combine():
    anno = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/labels/eye_contact_annotation.csv")
    anno = anno.rename(columns={'Unnamed: 0': 'frame'})
    recording_set = [i.split('.')[0] for i in anno.columns]
    recording_set.remove('frame')
    recording_set = sorted(list(set(recording_set)))
    anno_clean = pd.DataFrame()
    
    training_set = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_eye_contact.csv")
    training_set = sorted(set(training_set['recording']))
    
    testing_set = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_eye_contact.csv")
    testing_set = sorted(set(testing_set['recording']))
    
    all_set = training_set+testing_set
    
    for recording in all_set:
        temp_anno = anno.copy()
        temp_col = [col for col in temp_anno.columns if recording in col]
        temp_anno = temp_anno[['frame']+temp_col]
        temp_anno = temp_anno.rename(columns=temp_anno.iloc[0])
        temp_anno = temp_anno.rename(columns={temp_anno.columns[0]: 'frame'})
        temp_anno = temp_anno.drop(index=0)
        sub_col = temp_anno.columns[-4:]
        temp_anno = temp_anno.dropna(axis = 0, subset = sub_col, how = 'all')
        exist_member = temp_anno.columns[~temp_anno.isna().all()].tolist()
        temp_anno = temp_anno.fillna(0)
        temp_anno['frame'] = temp_anno['frame']+1
        
        for member in sub_col:
            if member in exist_member:
                temp_anno[member] = temp_anno[member].apply(get_direction, member=int(member.split('Pos')[1]))
            else:
                # pdb.set_trace()
                temp_anno[member] = [4 for i in range(len(temp_anno))]

        for member in range(1,5):
            op = get_op(recording, member)     
            temp_anno = temp_anno.merge(op, on='frame')
            
        temp_anno = temp_anno.astype(float)
        temp_anno['recording'] = recording
        anno_clean = pd.concat([anno_clean, temp_anno], axis=0)
    
    # pdb.set_trace()
    
    scaler = StandardScaler()
    output = pd.DataFrame(columns=['recording', 'member','label', 
                                   'gaze_angle_x', 'gaze_angle_y', 
                                   'head_angle_x', 'head_angle_y', 'head_angle_z',
                                   'head_pos_x', 'head_pos_y', 'head_pos_z', 
                                   'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                                   'gaze_1_x', 'gaze_1_y', 'gaze_1_z'])
    
    for ind, row in anno_clean.iterrows():
        for member in range(1,5):
            temp = pd.DataFrame(columns=['recording', 'member','label', 
                                         'gaze_angle_x', 'gaze_angle_y', 
                                         'head_angle_x', 'head_angle_y', 'head_angle_z',
                                         'head_pos_x', 'head_pos_y', 'head_pos_z', 
                                         'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                                         'gaze_1_x', 'gaze_1_y', 'gaze_1_z'])
            
            temp.loc[0, 'recording'] = row['recording']
            temp.loc[0, 'member'] = member
            temp.loc[0, 'label'] = row['subjectPos{}'.format(member)]
            temp.loc[0, 'gaze_angle_x'] = row['{}_gaze_angle_x'.format(member)]
            temp.loc[0, 'gaze_angle_y'] = row['{}_gaze_angle_y'.format(member)]
            temp.loc[0, 'head_angle_x'] = row['{}_head_x'.format(member)]
            temp.loc[0, 'head_angle_y'] = row['{}_head_y'.format(member)]
            temp.loc[0, 'head_angle_z'] = row['{}_head_z'.format(member)]
            temp.loc[0, 'head_pos_x'] = row['{}_pose_x'.format(member)]
            temp.loc[0, 'head_pos_y'] = row['{}_pose_y'.format(member)]
            temp.loc[0, 'head_pos_z'] = row['{}_pose_z'.format(member)]
            temp.loc[0, 'gaze_0_x'] = row['{}_gaze_0_x'.format(member)]
            temp.loc[0, 'gaze_0_y'] = row['{}_gaze_0_y'.format(member)]
            temp.loc[0, 'gaze_0_z'] = row['{}_gaze_0_z'.format(member)]
            temp.loc[0, 'gaze_1_x'] = row['{}_gaze_1_x'.format(member)]
            temp.loc[0, 'gaze_1_y'] = row['{}_gaze_1_y'.format(member)]
            temp.loc[0, 'gaze_1_z'] = row['{}_gaze_1_z'.format(member)]
            
            # pdb.set_trace()
            output = pd.concat([output, temp], axis=0)
    
    output = output.reset_index(drop=True)
    col = ['gaze_angle_x', 'gaze_angle_y', 
           'head_angle_x', 'head_angle_y', 'head_angle_z',
           'head_pos_x', 'head_pos_y', 'head_pos_z', 
           'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
           'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
    # output[['head_pos_x', 'head_pos_y', 'head_pos_z']] = scaler.fit_transform(output[['head_pos_x', 'head_pos_y', 'head_pos_z']].to_numpy())
    output[col] = scaler.fit_transform(output[col])
    
    joblib.dump(output, 'data/inference_4_class.pkl')
    pdb.set_trace()
    # joblib.dump(scaler, '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/standard_scaler.pkl')
   

# get_anno_multi_combine()


