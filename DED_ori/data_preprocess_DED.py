#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:29:56 2022

@author: shaohao
"""

import joblib
import pandas as pd
import numpy as np
import pdb
import os

sample_train = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
sample_val = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
sample = pd.concat([sample_train, sample_val]).reset_index(drop=True)

cv = 'GATE_other_2'
model_output = 'data/model_output/output_GATE_other_cv2.pkl'
# model_output = 'data/model_output/output_Combine_{}.pkl'.format(cv)
model_usage = os.path.basename(model_output).split('.')[0].split('_')[1:]
model_usage = '_'.join(model_usage)
fea_mode = 'indiv/{}'.format(cv)
output_csv = 'data/{}/DED_format.csv'.format(fea_mode)

output_dic = joblib.load(model_output)
group_lst = sorted(set([i.split('/')[-2] for i in output_dic.keys()]))

if not os.path.exists(output_csv):
    final_output = pd.DataFrame()
    
    for recording in group_lst:
        temp_sample = sample.copy()
        temp_sample = temp_sample[temp_sample['recording']==recording].reset_index(drop=True)
        temp_sample = temp_sample.drop(columns = ['start_time', 'end_time'])
    
        for ind in range(len(temp_sample)):
            for member in range(1,5):
                grab = '/homes/GPU2/shaohao/Corpus/multimediate/asd_multi_result/{}/row_{}_scores_{}.pckl'.format(recording, str(ind).zfill(3), member)
                pred = output_dic[grab]
                
                temp_sample.loc[ind, 'subjectPos{}'.format(member)] = pred
            
                
        final_output = pd.concat([final_output, temp_sample], axis=0)
    
    
    final_output = final_output.reset_index(drop=True)
    final_output.to_csv(output_csv, index=False)
# pdb.set_trace()
#%%
"""
model_output = pd.read_csv(output_csv)

# pdb.set_trace()
train_dialog = {}
train_talk_dict = {}
train_out_dict = {}

for recording in sorted(set(sample_train['recording'])):
    
    temp_tend = model_output.copy()
    temp_tend = temp_tend[temp_tend['recording']==recording]
    temp_tend = temp_tend.reset_index(drop=True)
    temp_tend['talk'] = temp_tend[['label_1','label_2','label_3','label_4']].apply(lambda x: list(x[x==1].index), axis=1)
    
    all_member = ['subjectPos{}'.format(i) for i in range(1,5)]    
    for member in range(1,5):
        other_member = ['subjectPos{}'.format(i) for i in range(1,5) if i != member]
        key = recording + '_' + str(member)
        recording_talk_sequence = [recording+'_'+str(member)+'_'+str(i).zfill(3) for i in range(len(temp_tend))]
        train_dialog[key] = recording_talk_sequence

        tend_copy = temp_tend.copy()
        talk_lst = list(tend_copy['talk'])
        talk_lst = ['silence' if len(i)==0 else i for i in talk_lst]
        talk_lst = ['me' if 'label_{}'.format(member) in i else i for i in talk_lst]
        talk_lst = ['other' if i not in ['silence', 'me'] else i for i in talk_lst]

        for ind, item in enumerate(recording_talk_sequence):
            # pdb.set_trace()
            train_talk_dict[item] = talk_lst[ind]
            # train_out_dict[item] = np.array([tend_copy.loc[ind,'subjectPos{}'.format(member)], 
            #                                  tend_copy.loc[ind, other_member].mean(), 
            #                                  tend_copy.loc[ind, all_member].mean()])
            
            me_talk = tend_copy.loc[ind,'subjectPos{}'.format(member)]
            temp_cal = 1
            for i in other_member:
                temp_cal *= (1-tend_copy.loc[ind,i])
            other_talk = (1.0-me_talk) * (1.0-temp_cal)
            all_silence = 1
            for i in all_member:
                all_silence *= (1-tend_copy.loc[ind,i])
            
            train_out_dict[item] = np.array([me_talk, other_talk, all_silence])
            

        # pdb.set_trace()
        
joblib.dump(train_out_dict, 'data/{}/me_other_train_prob.pkl'.format(fea_mode))  
joblib.dump(train_dialog, 'data/{}/me_other_train_dialog.pkl'.format(fea_mode))  
joblib.dump(train_talk_dict, 'data/{}/me_other_train_talk.pkl'.format(fea_mode))  
  
test_dialog = {}
test_talk_dict = {}
test_out_dict = {}

for recording in sorted(set(sample_val['recording'])):
    
    temp_tend = model_output.copy()
    temp_tend = temp_tend[temp_tend['recording']==recording]
    temp_tend = temp_tend.reset_index(drop=True)
    temp_tend['talk'] = temp_tend[['label_1','label_2','label_3','label_4']].apply(lambda x: list(x[x==1].index), axis=1)

    all_member = ['subjectPos{}'.format(i) for i in range(1,5)]
    for member in range(1,5):
        other_member = ['subjectPos{}'.format(i) for i in range(1,5) if i != member]
        key = recording + '_' + str(member)
        recording_talk_sequence = [recording+'_'+str(member)+'_'+str(i).zfill(3) for i in range(len(temp_tend))]
        test_dialog[key] = recording_talk_sequence
        
        tend_copy = temp_tend.copy()
        talk_lst = list(tend_copy['talk'])
        talk_lst = ['silence' if len(i)==0 else i for i in talk_lst]
        talk_lst = ['me' if 'label_{}'.format(member) in i else i for i in talk_lst]
        talk_lst = ['other' if i not in ['silence', 'me'] else i for i in talk_lst]
            
        for ind, item in enumerate(recording_talk_sequence):
            
            test_talk_dict[item] = talk_lst[ind]
            # test_out_dict[item] = np.array([tend_copy.loc[ind,'subjectPos{}'.format(member)], 
            #                                  tend_copy.loc[ind, other_member].mean(), 
            #                                  tend_copy.loc[ind, all_member].mean()])
            
            me_talk = tend_copy.loc[ind,'subjectPos{}'.format(member)]
            temp_cal = 1
            for i in other_member:
                temp_cal *= (1-tend_copy.loc[ind,i])
            other_talk = (1.0-me_talk) * (1.0-temp_cal)
            all_silence = 1
            for i in all_member:
                all_silence *= (1-tend_copy.loc[ind,i])
            
            test_out_dict[item] = np.array([me_talk, other_talk, all_silence])
        
joblib.dump(test_out_dict, 'data/{}/me_other_test_prob.pkl'.format(fea_mode))
joblib.dump(test_dialog, 'data/{}/me_other_test_dialog.pkl'.format(fea_mode))
joblib.dump(test_talk_dict, 'data/{}/me_other_test_talk.pkl'.format(fea_mode))
"""
#%%

model_output = pd.read_csv(output_csv)

train_df = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
# pdb.set_trace()
train_dialog = {}
train_talk_dict = {}
train_out_dict = {}

for recording in sorted(set(train_df['recording'])):
    
    temp_tend = model_output.copy()
    temp_tend = temp_tend[temp_tend['recording']==recording]
    # pdb.set_trace()
    for member in range(1,5):
        tend_copy = temp_tend.copy()
        tend_copy = tend_copy.reset_index(drop=True)
        key = recording + '_' + str(member)
        recording_talk_sequence = [recording+'_'+str(member)+'_'+str(i).zfill(3) for i in range(len(tend_copy))]
        train_dialog[key] = recording_talk_sequence
        
        tend_copy['talk'] = tend_copy['label_{}'.format(member)].apply(lambda x: 'talk' if x == 1 else 'silence')
        talk_lst = list(tend_copy['label_{}'.format(member)])
        # pdb.set_trace()
        for ind, item in enumerate(recording_talk_sequence):
            
            train_talk_dict[item] = talk_lst[ind]
            train_out_dict[item] = np.array([1-tend_copy.loc[ind,'subjectPos{}'.format(member)], tend_copy.loc[ind,'subjectPos{}'.format(member)]])
    
        
joblib.dump(train_out_dict, 'data/{}/indiv_train_prob.pkl'.format(fea_mode))  
joblib.dump(train_dialog, 'data/{}/indiv_train_dialog.pkl'.format(fea_mode))  
joblib.dump(train_talk_dict, 'data/{}/indiv_train_talk.pkl'.format(fea_mode))  
  
test_df = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
# pdb.set_trace()
test_dialog = {}
test_talk_dict = {}
test_out_dict = {}

for recording in sorted(set(test_df['recording'])):
    
    temp_tend = model_output.copy()
    temp_tend = temp_tend[temp_tend['recording']==recording]
    
    for member in range(1,5):
        tend_copy = temp_tend.copy()        
        tend_copy = tend_copy.reset_index(drop=True)

        key = recording + '_' + str(member)
        recording_talk_sequence = [recording+'_'+str(member)+'_'+str(i).zfill(3) for i in range(len(tend_copy))]
        test_dialog[key] = recording_talk_sequence
        
        tend_copy['talk'] = tend_copy['label_{}'.format(member)].apply(lambda x: 'talk' if x == 1 else 'silence')
        # talk_lst = list(tend_copy['label_{}'.format(member)])
        talk_lst = list(tend_copy['talk'])
        
        for ind, item in enumerate(recording_talk_sequence):
            
            test_talk_dict[item] = talk_lst[ind]
            test_out_dict[item] = np.array([1-tend_copy.loc[ind,'subjectPos{}'.format(member)], tend_copy.loc[ind,'subjectPos{}'.format(member)]])
    
        
joblib.dump(test_out_dict, 'data/{}/indiv_test_prob.pkl'.format(fea_mode))  
joblib.dump(test_dialog, 'data/{}/indiv_test_dialog.pkl'.format(fea_mode))  
joblib.dump(test_talk_dict, 'data/{}/indiv_test_talk.pkl'.format(fea_mode))  

#%%
"""
model_output = pd.read_csv(output_csv)

train_df = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)

train_dialog = {}
train_talk_dict = {}
train_out_dict = {}
'''
re-order
    label 1 -> talk most
    label 4 -> talk least
'''
for recording in sorted(set(train_df['recording'])):
    
    temp_tend = model_output.copy()
    temp_tend = temp_tend[temp_tend['recording']==recording]
    temp_tend_sum = temp_tend.sum()[-4:]
    temp_tend_sum = pd.to_numeric(temp_tend_sum)
    temp_tend_sort = list(temp_tend_sum.sort_values(ascending=False).index)
    talk_0 = temp_tend_sort[0]
    talk_1 = temp_tend_sort[1]
    talk_2 = temp_tend_sort[2]
    talk_3 = temp_tend_sort[3]
    
    recording_tend = temp_tend[[talk_0, talk_1,talk_2,talk_3]]
    recording_tend = recording_tend.rename(columns={talk_0: 'most_speaker', talk_1: 'second_speaker',
                                                    talk_2: 'third_speaker', talk_3: 'least_speaker'})
    
    
    df_col = [i.replace('subjectPos', 'label_') for i in temp_tend_sort]    
    temp_df = train_df.copy()
    temp_df = temp_df[temp_df['recording']==recording]
    # import pdb;pdb.set_trace()
    temp_df = temp_df[['recording', 'start_time', 'end_time', df_col[0], df_col[1], df_col[2], df_col[3]]]
    temp_df = temp_df.rename(columns={df_col[0]: 'most_speaker', df_col[1]: 'second_speaker',
                                      df_col[2]: 'third_speaker', df_col[3]: 'least_speaker'})
    temp_df['talk'] = temp_df[['most_speaker','second_speaker','third_speaker','least_speaker']].apply(lambda x: list(x[x==1].index), axis=1)
    talk_lst = list(temp_df['talk'])
    recording_talk_sequence = [recording+'_'+str(i).zfill(3) for i in range(len(temp_df))]
    train_dialog[recording] = recording_talk_sequence
    
    assert len(talk_lst) == len(recording_talk_sequence)
    assert len(talk_lst) == len(recording_tend)
    
    for ind, item in enumerate(recording_talk_sequence):
        train_talk_dict[item] = talk_lst[ind]
        train_out_dict[item] = recording_tend.iloc[ind].values.astype(float)
    
        
joblib.dump(train_out_dict, 'data/{}/character_train_prob.pkl'.format(fea_mode))  
joblib.dump(train_dialog, 'data/{}/character_train_dialog.pkl'.format(fea_mode))  
joblib.dump(train_talk_dict, 'data/{}/character_train_talk.pkl'.format(fea_mode))  
  
test_df = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)

test_dialog = {}
test_talk_dict = {}
test_out_dict = {}
'''
re-order
    label 1 -> talk most
    label 4 -> talk least
'''
for recording in sorted(set(test_df['recording'])):
    
    temp_tend = model_output.copy()
    temp_tend = temp_tend[temp_tend['recording']==recording]
    temp_tend_sum = temp_tend.sum()[-4:]
    temp_tend_sum = pd.to_numeric(temp_tend_sum)
    temp_tend_sort = list(temp_tend_sum.sort_values(ascending=False).index)
    talk_0 = temp_tend_sort[0]
    talk_1 = temp_tend_sort[1]
    talk_2 = temp_tend_sort[2]
    talk_3 = temp_tend_sort[3]
    
    recording_tend = temp_tend[[talk_0, talk_1,talk_2,talk_3]]
    recording_tend = recording_tend.rename(columns={talk_0: 'most_speaker', talk_1: 'second_speaker',
                                                    talk_2: 'third_speaker', talk_3: 'least_speaker'})
    
    
    df_col = [i.replace('subjectPos', 'label_') for i in temp_tend_sort]    
    temp_df = test_df.copy()
    temp_df = temp_df[temp_df['recording']==recording]
    # import pdb;pdb.set_trace()
    temp_df = temp_df[['recording', 'start_time', 'end_time', df_col[0], df_col[1], df_col[2], df_col[3]]]
    temp_df = temp_df.rename(columns={df_col[0]: 'most_speaker', df_col[1]: 'second_speaker',
                                      df_col[2]: 'third_speaker', df_col[3]: 'least_speaker'})
    temp_df['talk'] = temp_df[['most_speaker','second_speaker','third_speaker','least_speaker']].apply(lambda x: list(x[x==1].index), axis=1)
    talk_lst = list(temp_df['talk'])
    recording_talk_sequence = [recording+'_'+str(i).zfill(3) for i in range(len(temp_df))]
    test_dialog[recording] = recording_talk_sequence
    
    assert len(talk_lst) == len(recording_talk_sequence)
    assert len(talk_lst) == len(recording_tend)
    
    for ind, item in enumerate(recording_talk_sequence):
        test_talk_dict[item] = talk_lst[ind]
        test_out_dict[item] = recording_tend.iloc[ind].values.astype(float)
    
        
joblib.dump(test_out_dict, 'data/{}/character_test_prob.pkl'.format(fea_mode))  
joblib.dump(test_dialog, 'data/{}/character_test_dialog.pkl'.format(fea_mode))  
joblib.dump(test_talk_dict, 'data/{}/character_test_talk.pkl'.format(fea_mode)) 

"""