#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 23:59:01 2022

@author: shaohao
"""


from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch import nn

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score

import os
import torch
import random
import numpy as np
import joblib
import pdb
import tqdm
import pandas as pd
from glob import glob

def display(bi_acc, f1, recall, precision, uar):
    print('='*80)
    print("Binary accuracy on test set is {}".format(bi_acc))
    print("F1-score on test set is {}".format(f1))
    print("Recall on test set is {}".format(recall))
    print("Precision on test set is {}".format(precision))
    print("UAR on test set is {}".format(uar))
    
def get_id(member, direction):
 
    if direction == 0:
        return 0 
    elif direction == 4:
        pdb.set_trace()
    else:
        temp = member+direction
        if temp > 4:
            temp = temp-4
        return temp

def get_id_lst(member, direction):
    
    output = []
    for x in direction:
        if x == 0:
            output.append(x)
        elif x == 4:
            pdb.set_trace()
        else:
            temp = member+x
            if temp > 4:
                temp = temp-4
            output.append(temp)
    return output

OUTPUT_DIM = 4
RANDSEED = 2021
np.random.seed(RANDSEED)
torch.manual_seed(RANDSEED)
torch.cuda.manual_seed(RANDSEED)
torch.cuda.manual_seed_all(RANDSEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(RANDSEED)
random.seed(RANDSEED)

recording_lst = sorted(glob('/homes/GPU2/shaohao/Corpus/multimediate/openface_result/*'))
recording_lst = [i.split('/')[-1] for i in recording_lst]

# pdb.set_trace()
FEATURE = [' gaze_angle_x',' gaze_angle_y',' pose_Rx',' pose_Ry',' pose_Rz',
           ' gaze_0_x',' gaze_0_y',' gaze_0_z',' gaze_1_x',' gaze_1_y',' gaze_1_z']

model_path = "/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/gaze_model/best_model/BEST_4_class_no_std.sav"
DTYPE = torch.cuda.FloatTensor
#%%
# print('Inference all files. . . ')
 
# model = torch.load(model_path)
# # scaler = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/standardscaler.pkl")
# model.eval()
# for recording in tqdm.tqdm(recording_lst):
#     for member in range(1,5):
#         output = pd.DataFrame(columns = ['recording', 'frame', 
#                                          'member_1_pred_dir', 'member_1_pred_id',
#                                          'member_2_pred_dir', 'member_2_pred_id',
#                                          'member_3_pred_dir', 'member_3_pred_id',
#                                          'member_4_pred_dir', 'member_4_pred_id',])
        
#         op = pd.read_csv('/homes/GPU2/shaohao/Corpus/multimediate/openface_result/{}/subjectPos{}.video.csv'.format(recording, member))
#         op = op[['frame', ' pose_Rx', ' pose_Ry', ' pose_Rz', ' gaze_angle_x', ' gaze_angle_y',
#                   ' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z']]
            
#         x = op[FEATURE].values.astype(float)

#         x = torch.from_numpy(x)
#         # pdb.set_trace()
#         x = Variable(x.float().type(DTYPE), requires_grad=False)
#         output_test = model(x)
#         soft = nn.Softmax(dim=1)
#         output_test = soft(output_test)
#         output_test = output_test.cpu().data.numpy().reshape(-1, OUTPUT_DIM)
    
#         output_test = np.argmax(output_test, axis=-1)       #dir
        
#         output['recording'] = recording
#         output['frame'] = op['frame']
#         output['member_{}_pred_dir'.format(member)] = output_test
#         # pdb.set_trace()
#         output['member_{}_pred_id'.format(member)] = get_id_lst(member, output_test)
        
#         output.to_csv('/homes/GPU2/shaohao/Corpus/multimediate/gaze_model_inference/{}/inference_result/{}_inference.csv'.format(recording, member))

#%%
# print('verify the performance . . . ')

# anno = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/labels/eye_contact_annotation.csv")
# anno = anno.rename(columns={'Unnamed: 0': 'frame'})
# recording_set = [i.split('.')[0] for i in anno.columns]
# recording_set.remove('frame')
# recording_set = sorted(list(set(recording_set)))

# gt = []
# pred = []
# for recording in recording_set:
#     temp_anno = anno.copy()
#     temp_col = [col for col in temp_anno.columns if recording in col]
#     temp_anno = temp_anno[['frame']+temp_col]
#     temp_anno = temp_anno.rename(columns=temp_anno.iloc[0])
#     temp_anno = temp_anno.rename(columns={temp_anno.columns[0]: 'frame'})
#     temp_anno = temp_anno.drop(index=0)
#     sub_col = temp_anno.columns[-4:]
#     temp_anno = temp_anno.dropna(axis = 0, subset = sub_col, how = 'all')
#     lack_member = temp_anno.columns[temp_anno.isna().all()].tolist()
#     exist_member = temp_anno.columns[~temp_anno.isna().all()].tolist()
#     exist_member.remove('frame')
#     temp_anno = temp_anno.fillna(0)
#     temp_anno['frame'] += 1
    
#     inference = pd.read_csv('/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/inference_result/{}_inference.csv'.format(recording), index_col=0)
#     infer_align = temp_anno.merge(inference, on='frame')
#     # pdb.set_trace()
#     # for i in list(infer_align.columns[1:5]):
#     #     member = i.split('Pos')[-1]
#     #     if len(lack_member)>0:
#     #         lack = int(lack_member[0].split('Pos')[-1])
#     #     if i not in exist_member:
#     #         infer_align['member_{}_pred_id'.format(member)] = 0
#     #         # print( infer_align['member_{}_pred_id'.format(member)])
#     #         continue
#     #     infer_align['member_{}_pred_id'.format(member)] =  infer_align['member_{}_pred_id'.format(member)].apply(lambda x: x if x!=lack else 0)
    
#     # pdb.set_trace()
#     for i in range(1,5):
        
#         gt.extend(list(infer_align['subjectPos{}'.format(i)].astype(float).astype(int)))
#         pred.extend(list(infer_align['member_{}_pred_id'.format(i)].astype(float).astype(int)))
        
        
# bi_acc = accuracy_score(gt, pred)
# f1 = f1_score(gt, pred, average='micro')
# precision = precision_score(gt, pred, average='macro', zero_division=0)
# recall = recall_score(gt, pred, average='macro', zero_division=0)
# uar = (precision+recall)/2.0

# print("Binary accuracy on test set is {:.4f}".format(bi_acc))
# print("F1-score on test set is {:.4f}".format(f1))
# print("Recall on test set is {:.4f}".format(recall))
# print("Precision on test set is {:.4f}".format(precision))
# print("UAR on test set is {:.4f}".format(uar))

# # print(metrics.confusion_matrix(gt, pred))
# print(metrics.classification_report(gt, pred, digits=3, zero_division=0))
#%%
print('Inference all files. . . ')
 
model = torch.load(model_path)
# scaler = joblib.load("/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/eye_contact/data/standardscaler.pkl")
model.eval()
for recording in tqdm.tqdm(recording_lst):
    if not os.path.exists('/homes/GPU2/shaohao/Corpus/multimediate/gaze_model_inference/{}'.format(recording)):
        os.mkdir('/homes/GPU2/shaohao/Corpus/multimediate/gaze_model_inference/{}'.format(recording))
    for member in range(1,5):
        output = pd.DataFrame(columns = ['recording', 'frame', 'look_empty_prob', 'look_right_prob', 'look_middle_prob', 'look_left_prob'])
        op = pd.read_csv('/homes/GPU2/shaohao/Corpus/multimediate/openface_result/{}/subjectPos{}.video.csv'.format(recording, member))
        op = op[['frame', ' pose_Rx', ' pose_Ry', ' pose_Rz', ' gaze_angle_x', ' gaze_angle_y',
                  ' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z']]
            
        x = op[FEATURE].values.astype(float)

        x = torch.from_numpy(x)
        # pdb.set_trace()
        x = Variable(x.float().type(DTYPE), requires_grad=False)
        output_test = model(x)
        soft = nn.Softmax(dim=1)
        output_test = soft(output_test)
        output_test = output_test.cpu().data.numpy().reshape(-1, OUTPUT_DIM)
    
        output['recording'] = [recording for i in range(output_test.shape[0])]
        output['frame'] = op['frame']
        output['look_empty_prob'] = output_test[:, 0]
        output['look_right_prob'] = output_test[:, 1]
        output['look_middle_prob'] = output_test[:, 2]
        output['look_left_prob'] = output_test[:, 3]
        assert output.isnull().sum().sum() == 0
        
        output.to_csv('/homes/GPU2/shaohao/Corpus/multimediate/gaze_model_inference/{}/{}_inference.csv'.format(recording, member))

#%%
import warnings
warnings.filterwarnings("ignore")
print('inference on train test data . . . ')
  
train_data = joblib.load('data/all_group/four_class/multi_gaze_training_stretch_all_group.pkl')
test_data = joblib.load('data/all_group/four_class/multi_gaze_testing_stretch_all_group.pkl')
all_data = pd.concat([train_data, test_data])

all_recording = sorted(set(train_data['recording']))+sorted(set(test_data['recording']))
# pdb.set_trace()
model_path = "/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/gaze_model/best_model/BEST_4_class_no_std.sav"

DTYPE = torch.cuda.FloatTensor
  
model = torch.load(model_path)
FEATURE = ['gaze_angle_x', 'gaze_angle_y','head_angle_x', 'head_angle_y', 'head_angle_z',
           'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
# data_set = multi(all_data, FEATURE)
# data_iterator = DataLoader(data_set, batch_size=len(data_set), num_workers=4, shuffle=True)
 
model = torch.load(model_path)
model.eval()
x = all_data[FEATURE].values.astype(float)
x = torch.from_numpy(x)
y = all_data['label'].values.astype(float)
y = torch.from_numpy(y)

x = Variable(x.float().type(DTYPE), requires_grad=False)
y = Variable(y.float().type(torch.cuda.LongTensor), requires_grad=False)
output_test = model(x)
soft = nn.Softmax(dim=1)
output_test = soft(output_test)

output_test = output_test.cpu().data.numpy().reshape(-1, OUTPUT_DIM)
y = y.cpu().data.numpy()

# these are the needed metrics
output_test = np.argmax(output_test, axis=-1)


bi_acc = accuracy_score(y, output_test)
f1 = f1_score(y, output_test, average='macro')
precision = precision_score(y, output_test, average='macro', zero_division=0)
recall = recall_score(y, output_test, average='macro')
uar = (precision+recall)/2.0
display(bi_acc, f1, recall, precision, uar)
print('='*80)
print('\n')

print(metrics.classification_report(y, output_test, digits=3, zero_division=0))

#%%
print('inference on train test data . . . ')
  
train_data = joblib.load('data/all_group/four_class/multi_gaze_training_stretch_all_group_.pkl')
test_data = joblib.load('data/all_group/four_class/multi_gaze_testing_stretch_all_group_.pkl')
all_data = pd.concat([train_data, test_data]).reset_index(drop=True)

all_recording = sorted(set(train_data['recording']))+sorted(set(test_data['recording']))
# pdb.set_trace()
model_path = "/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/gaze_model/best_model/BEST_4_class_no_std.sav"

DTYPE = torch.cuda.FloatTensor
  
model = torch.load(model_path)
FEATURE = ['gaze_angle_x', 'gaze_angle_y','head_angle_x', 'head_angle_y', 'head_angle_z',
           'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z']
# data_set = multi(all_data, FEATURE)
# data_iterator = DataLoader(data_set, batch_size=len(data_set), num_workers=4, shuffle=True)
 
model = torch.load(model_path)
model.eval()
for ind, row in all_data.iterrows():
    x = row[FEATURE].values.astype(float)
    # pdb.set_trace()
    x = torch.from_numpy(x).unsqueeze(dim=0)
    y = all_data['dir_label'].values.astype(float)
    y = torch.from_numpy(y).unsqueeze(dim=0)
    
    x = Variable(x.float().type(DTYPE), requires_grad=False)
    y = Variable(y.float().type(torch.cuda.LongTensor), requires_grad=False)
    output_test = model(x)
    
    output_test = output_test.cpu().data.numpy().reshape(-1, OUTPUT_DIM)
    y = y.cpu().data.numpy()
    
    # these are the needed metrics
    output_test = np.argmax(output_test, axis=-1)
    
    all_data.loc[ind, 'dir_pred'] = output_test[0]
    # pdb.set_trace()

# pdb.set_trace()
bi_acc = accuracy_score(all_data['dir_label'].values.astype(float), all_data['dir_pred'].values.astype(float))
f1 = f1_score(all_data['dir_label'].values.astype(float), all_data['dir_pred'].values.astype(float), average='macro')
precision = precision_score(all_data['dir_label'].values.astype(float), all_data['dir_pred'].values.astype(float), average='macro', zero_division=0)
recall = recall_score(all_data['dir_label'].values.astype(float), all_data['dir_pred'].values.astype(float), average='macro')
uar = (precision+recall)/2.0
display(bi_acc, f1, recall, precision, uar)
print('='*80)
print('\n')

print(metrics.classification_report(all_data['dir_label'].values.astype(float), all_data['dir_pred'].values.astype(float), digits=3, zero_division=0))

all_data['id_pred'] = all_data.apply(lambda x: get_id(x['member'], x['dir_pred']), axis=1)
bi_acc = accuracy_score(all_data['id_label'].values.astype(float), all_data['id_pred'].values.astype(float))
f1 = f1_score(all_data['id_label'].values.astype(float), all_data['id_pred'].values.astype(float), average='macro')
precision = precision_score(all_data['id_label'].values.astype(float), all_data['id_pred'].values.astype(float), average='macro', zero_division=0)
recall = recall_score(all_data['id_label'].values.astype(float), all_data['id_pred'].values.astype(float), average='macro')
uar = (precision+recall)/2.0
display(bi_acc, f1, recall, precision, uar)
print('='*80)
print('\n')

print(metrics.classification_report(all_data['id_label'].values.astype(float), all_data['id_pred'].values.astype(float), digits=3, zero_division=0))

# all_data.to_csv('./GOOD.csv')


