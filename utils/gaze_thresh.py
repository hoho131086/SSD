#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:02:45 2022

@author: shaohao
"""
import tqdm
import pandas as pd
import numpy as np
from glob import glob
import math
import pdb
import warnings
import scipy
import scipy.spatial
import scipy.stats
warnings.simplefilter("ignore")

def turn_frame(df):
    df['start_time'] = int(np.round(df['start_time']/(1/30)))
    df['end_time'] = int(np.round(df['end_time']/(1/30)))
    return df

def get_openface_df(recording, drop_member):
    group_fea = {}
    openface_dir = '/homes/GPU2/shaohao/Corpus/multimediate/openface_result/'
    used_col = ['frame', ' gaze_angle_x', ' gaze_angle_y']
    for i in range(1,5):
        if i in drop_member: continue
        temp = pd.read_csv(glob(openface_dir+recording+'/*{}.video.csv'.format(i))[0])
        # pdb.set_trace()
        temp = temp[used_col]
        group_fea[i] = temp
    
    return group_fea
    # return 0

def get_angle(group_fea, row, current): # current = candidate

    current_df = group_fea[current]
    current_df = current_df.iloc[row['start_time']:row['end_time']]
    # import pdb;pdb.set_trace()
    temp = 0
    for key in group_fea.keys():
        if key == current: continue
        key_df = group_fea[key]
        key_df = key_df.iloc[row['start_time']:row['end_time']]
        
        if current-key == 1 or current-key == -3:
            rad_1 = math.radians(-45)
            temp_output = key_df.apply(lambda x: 1-np.abs(x[' gaze_angle_x']-rad_1), axis=1)
            
        elif current-key == 2 or current-key == -2:
            # pdb.set_trace()
            rad_1 = math.radians(0)
            temp_output = key_df.apply(lambda x: 1-np.abs(x[' gaze_angle_x']-rad_1), axis=1)
            
        elif current-key == -1 or current-key == 3:
            rad_1 = math.radians(45)
            temp_output = key_df.apply(lambda x: 1-np.abs(x[' gaze_angle_x']-rad_1), axis=1)
        else:
            pdb.set_trace()
        
        # pdb.set_trace()
        # if temp_output.shape!=(300,):
        #     pdb.set_trace()
        if len(temp_output)!=0:
            temp+=temp_output.mean()
        else:
            pdb.set_trace()
    
    return temp/(len(group_fea.keys())-1)

def cos_sim(df):
    label_col = list(df.index)[-8:-4]
    anno_col = list(df.index)[-4:]
    # pdb.set_trace()
    return 1 - scipy.spatial.distance.cosine(df[label_col].values.astype(np.float), df[anno_col].values.astype(np.float))

def pearson(df):
    label_col = list(df.index)[-8:-4]
    anno_col = list(df.index)[-4:]
    # pdb.set_trace()
    return scipy.stats.pearsonr(df[label_col].values.astype(np.float), df[anno_col].values.astype(np.float))

#%%
sample_df = pd.DataFrame()
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
temp_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
sample_df = pd.concat([sample_df, temp_sample], axis=0)
sample_df = sample_df.reset_index(drop=True)
sample_df = sample_df.apply(turn_frame, axis=1)

annotation_file = "/homes/GPU2/shaohao/Corpus/multimediate/labels/next_speaker_annotation.csv"
anno_df = pd.read_csv(annotation_file, index_col=0)   
anno_df = anno_df.T
anno_df = anno_df.reset_index()
anno_df = anno_df.rename(columns={'index': 'recording', anno_df.columns[1]: 'subject'})
anno_df['recording'] = anno_df['recording'].apply(lambda x: x.split('.')[0])
anno_df['subject'] = anno_df['subject'].apply(lambda x: int(x.split('Pos')[1]))

#%%
all_member = [1,2,3,4]

gaze = pd.DataFrame(columns=['subject1', 'subject2', 'subject3', 'subject4'])

# sample_df = sample_df.drop_duplicates(subset=['recording'])

for ind, row in tqdm.tqdm(sample_df.iterrows()):
    # if ind < 7185: continue
    recording = row['recording']
    # if recording == 'recording25' or recording == 'recording11':continue
    member_check = anno_df[anno_df['recording']==recording]
    member_check = member_check[member_check[0.0].notna()]
    member_check = list(member_check['subject'])
    drop_member = [i for i in all_member if i not in member_check]
    group_fea = get_openface_df(recording, drop_member)
    for member in all_member:
        if member in drop_member:
            gaze.loc[ind, 'subject{}'.format(member)]=0
            continue
        gaze.loc[ind, 'subject{}'.format(member)]=get_angle(group_fea, row, member)
    pdb.set_trace()
    # print(gaze)
    # gaze.to_csv('check.csv')

merge_df = pd.concat([sample_df, gaze], axis=1)
merge_df.to_csv('merge_new.csv')
  
# pdb.set_trace()
#%% correlation + similarity
output_sim = merge_df.apply(cos_sim, axis=1)
output_sim = output_sim.dropna()

pear = merge_df.apply(pearson, axis=1)
pear = pd.DataFrame(pear, columns=['tuple'])
pear['correlation'], pear['P_value'] = zip(*pear['tuple'])
pear = pear.dropna()
pear = pear.drop(columns=['tuple'])

print(output_sim.mean())
print(pear.mean())

#%%
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
LABEL_COL=['label_1', 'label_2', 'label_3', 'label_4']
PRED_COL=['subject1', 'subject2', 'subject3', 'subject4']


def thres(df, th):
    if df['subject1'] >= th:
        df['subject1'] = 1
    else:
        df['subject1'] = 0
    
    if df['subject2'] >= th:
        df['subject2'] = 1
    else:
        df['subject2'] = 0
        
    if df['subject3'] >= th:
        df['subject3'] = 1
    else:
        df['subject3'] = 0
        
    if df['subject4'] >= th:
        df['subject4'] = 1
    else:
        df['subject4'] = 0
        
    return df


precision_lst=[]
recall_lst=[]
uar_lst=[]

for thresh_var in range(0, 100, 5):
    if thresh_var!=30:continue
    thresh_df = merge_df.copy()
    thresh_df = thresh_df.apply(thres, th = (thresh_var/100.0), axis=1)
    iden = pd.DataFrame(thresh_df[LABEL_COL].values==thresh_df[PRED_COL].values, columns=['pred_1', 'pred_2',
                                                                                          'pred_3', 'pred_4'])
    iden = pd.concat([thresh_df, iden], axis=1)
    iden[list(iden.columns[-4:])] = iden[list(iden.columns[-4:])].astype(int)
    gt = np.hstack([thresh_df['label_1'].values, thresh_df['label_2'].values, 
                    thresh_df['label_3'].values, thresh_df['label_4'].values])
    pred =  np.hstack([thresh_df['subject1'].values, thresh_df['subject2'].values, 
                    thresh_df['subject3'].values, thresh_df['subject4'].values])
    
    precision = precision_score(gt, pred, average='macro')
    recall = recall_score(gt, pred, average='macro')
    uar = (precision+recall)/2.0
    precision_lst.append(precision)
    recall_lst.append(recall)
    uar_lst.append(uar)
    print('thresh hold: {:.2f}, precision: {:.4f}, recall: {:.4f}, uar: {:.4f}'
          .format(thresh_var/100.0, precision, recall, uar))

result = pd.DataFrame({'precision': precision_lst, 'recall':recall_lst, 'uar': uar_lst})
# result.to_csv('gaze_result.csv')











    





