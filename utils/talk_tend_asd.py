#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:26:45 2022

@author: shaohao
"""

import joblib
from glob import glob
import os
import pandas as pd
import pdb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import scipy.interpolate as interp

def get_data(root, recording, ind, member, lack_member):
    data_path = root+recording+'/row_{}_scores_{}.pckl'.format(str(ind).zfill(3), member)
    if not os.path.exists(data_path):
        pdb.set_trace()
    data = joblib.load(data_path)
    
    if len(data)!= 0:
        asd_ind = max((len(l), i) for i, l in enumerate(data))[1]
    else:
        asd_ind = 0
        data = [[-3 for i in range(249)]]
    if len(data[asd_ind])!=249:
        data = np.hstack([data[asd_ind],np.array([-3 for i in range(249)])])
        data = data[:249]
    else:
        data = data[asd_ind]
        
    data = np.array(data)
    interp_talk = interp.interp1d(np.arange(data.size),data)
    talk_file_stretch = interp_talk(np.linspace(0,data.size-1,300))
    
    return ((talk_file_stretch >= 0).sum())/250
    # if len(data) == 0:
    #     data = [np.array([-1 for i in range(249)])]
    
    # try:
    #     asd_ind = max((len(l), i) for i, l in enumerate(data))[1]     
    # except:
    #     print(data_path)
    
    # if len(data[asd_ind])!=249:
    #     try:
    #         asd_result = np.hstack((data[asd_ind],np.array([-1 for i in range(249-len(data[asd_ind]))])))
    #     except:
    #         pdb.set_trace()
    # else:
    #     asd_result = data[asd_ind]
    
    # if len(asd_result)!=249:
    #     pdb.set_trace()
    
    # return ((asd_result >= 0).sum())/250
    # return asd_result

def thres(df, th):
    if df['subjectPos1'] >= th:
        df['subjectPos1'] = 1
    else:
        df['subjectPos1'] = 0
    
    if df['subjectPos2'] >= th:
        df['subjectPos2'] = 1
    else:
        df['subjectPos2'] = 0
        
    if df['subjectPos3'] >= th:
        df['subjectPos3'] = 1
    else:
        df['subjectPos3'] = 0
        
    if df['subjectPos4'] >= th:
        df['subjectPos4'] = 1
    else:
        df['subjectPos4'] = 0
        
    return df

def thres_2(df, th):
    if df['value'] >= th:
        df['value'] = 1
    else:
        df['value'] = 0
    
        
    return df

#%%
ROOT = '/homes/GPU2/shaohao/turn_taking/turn_changing/active_speaker/asd_multi_result/'
train_df = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
test_df = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
test_recording = sorted(set(test_df.recording))
all_df = pd.concat([train_df, test_df])

recording_set = glob(ROOT+'*')
recording_set = sorted([os.path.basename(i) for i in recording_set])
recording_set.remove('recording25_old')

annotation_file = "/homes/GPU2/shaohao/Corpus/multimediate/labels/next_speaker_annotation.csv"
anno_df = pd.read_csv(annotation_file, index_col=0)   
anno_df = anno_df.T
anno_df = anno_df.reset_index()
anno_df = anno_df.rename(columns={'index': 'recording', anno_df.columns[1]: 'subject'})
anno_df['recording'] = anno_df['recording'].apply(lambda x: x.split('.')[0])
anno_df['subject'] = anno_df['subject'].apply(lambda x: int(x.split('Pos')[1]))

lack_member = {}
for recording in recording_set:
    temp_anno = anno_df.copy()
    temp_anno = temp_anno[temp_anno['recording']==recording].T
    temp_anno = temp_anno.rename(columns=temp_anno.iloc[1])
    temp_anno = temp_anno.iloc[2:]
    temp_anno = temp_anno.dropna(axis = 0, how = 'all')
    if len(temp_anno.columns[temp_anno.isna().all()].tolist())>0:
        lack_member[recording] = temp_anno.columns[temp_anno.isna().all()].tolist()[0]

talk_df = pd.DataFrame()
for recording in recording_set:
    out = pd.DataFrame(columns = ['recording', 'start_time', 'subjectPos1', 'subjectPos2', 'subjectPos3', 'subjectPos4',
                                  'label_1', 'label_2', 'label_3', 'label_4'])
    temp_time = all_df.copy()
    temp_time = temp_time[temp_time['recording']==recording].reset_index(drop=True)
    
    for ind, row in temp_time.iterrows():
        for member in range(1,5):
            data = get_data(ROOT, recording, ind, member, lack_member)
            out.loc[ind, 'subjectPos{}'.format(member)] = data
            out.loc[ind, 'label_{}'.format(member)] = row['label_{}'.format(member)]
            out.loc[ind, 'recording'] = recording
            out.loc[ind, 'start_time'] = row['start_time']
    print(recording)
    print(out.head())    
    # out.to_csv('ASD_talk_tend/{}.csv'.format(recording))
    talk_df = pd.concat([talk_df, out])

# pdb.set_trace()
#%%
precision_lst=[]
f1_lst=[]
uar_lst=[]
acc_lst = []

talk_2_df = talk_df.copy()
talk_2_df = talk_2_df[['recording', 'start_time', 'subjectPos1', 'subjectPos2', 'subjectPos3','subjectPos4', 'label_1', 'label_2', 'label_3', 'label_4']]      
talk_2_df = pd.melt(talk_2_df, id_vars='recording', value_vars=['subjectPos1', 'subjectPos2', 'subjectPos3','subjectPos4', 'label_1', 'label_2', 'label_3', 'label_4'])
talk_2_df['value'] = pd.to_numeric(talk_2_df['value'])

trans_dic = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/final/DED/idx/transit.pkl")
for thresh_var in range(0, 105, 10):
    gt_s = []
    pred_s = []
    gt_t = []
    pred_t = []
    gt_all = []
    pred_all = []

    for r in test_recording:
        for i in range(1,5):
            thresh_df = talk_2_df.copy()
            thresh_df = thresh_df[thresh_df['recording']==r]
            
            pred_df = list(thresh_df[thresh_df['variable']=='subjectPos{}'.format(i)]['value'])
            pred_df = [1 if K >= (thresh_var/100.0) else 0 for K in pred_df]
            label_df = list(thresh_df[thresh_df['variable']=='label_{}'.format(i)]['value'])
            
            trans_idx = trans_dic[r+'_'+str(i)]
            
            gt_s.extend([prob for K, prob in enumerate(label_df) if K not in trans_idx])
            gt_t.extend([prob for K, prob in enumerate(label_df) if K in trans_idx])
            gt_all.extend(label_df)
            
            pred_s.extend([prob for K, prob in enumerate(pred_df) if K not in trans_idx])
            pred_t.extend([prob for K, prob in enumerate(pred_df) if K in trans_idx])
            pred_all.extend(pred_df)

    # precision = precision_score(gt_t, pred_t, average='macro')
    # uar = recall_score(gt_t, pred_t, average='macro')
    # f1 = f1_score(gt_t, pred_t, average='macro')
    # acc = accuracy_score(gt_t, pred_t)
    # print('TRANS thresh hold: {:.2f}, uar: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}'.format(thresh_var/100.0, uar, acc, f1, precision))

    # precision = precision_score(gt_s, pred_s, average='macro')
    # uar = recall_score(gt_s, pred_s, average='macro')
    # f1 = f1_score(gt_s, pred_s, average='macro')
    # acc = accuracy_score(gt_s, pred_s)
    # print('SAEM thresh hold: {:.2f}, uar: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}'.format(thresh_var/100.0, uar, acc, f1, precision))

    precision = precision_score(gt_all, pred_all, average='macro')
    uar = recall_score(gt_all, pred_all, average='macro')
    f1 = f1_score(gt_all, pred_all, average='macro')
    acc = accuracy_score(gt_all, pred_all)
    print('ALL thresh hold: {:.2f}, uar: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}'.format(thresh_var/100.0, uar, acc, f1, precision))

# result = pd.DataFrame({'precision': precision_lst, 'uar':uar_lst, 'f1': f1_lst, 'acc': acc_lst})

# for thresh_var in range(0, 105, 5):
#     thresh_df = talk_df.copy()
#     thresh_df = thresh_df[thresh_df['recording'].isin(test_recording)]
#     thresh_df = thresh_df.apply(thres, th = (thresh_var/100.0), axis=1)
#     gt = np.hstack([thresh_df['label_1'].values, thresh_df['label_2'].values, 
#                     thresh_df['label_3'].values, thresh_df['label_4'].values])
#     pred =  np.hstack([thresh_df['subjectPos1'].values, thresh_df['subjectPos2'].values, 
#                     thresh_df['subjectPos3'].values, thresh_df['subjectPos4'].values])
    
#     precision = precision_score(gt, pred, average='macro')
#     uar = recall_score(gt, pred, average='macro')
#     f1 = f1_score(gt, pred, average='macro')
#     acc = accuracy_score(gt, pred)
#     precision_lst.append(precision)
#     f1_lst.append(f1)
#     uar_lst.append(uar)
#     acc_lst.append(acc)
#     print('thresh hold: {:.2f}, precision: {:.4f}, uar: {:.4f}, f1: {:.4f}, acc: {:.4f}'
#           .format(thresh_var/100.0, precision, uar, f1, acc))

# result = pd.DataFrame({'precision': precision_lst, 'uar':uar_lst, 'f1': f1_lst, 'acc': acc_lst})
    
# result.to_csv('talk_result_multi_asd_reslt.csv')



