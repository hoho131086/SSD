#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 11:53:46 2022

@author: shaohao
"""


import numpy as np
import pandas as pd
import joblib
from sklearn import tree
import graphviz 
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


ADD_EGE = False
ADD_OPENFACE = False

    

#%% data preprocess
train_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/train_next_speaker.csv", index_col=0)
train_sample = train_sample.reset_index(drop=True)
train_record = sorted(list(set(train_sample['recording'])))
valid_sample = pd.read_csv("/homes/GPU2/shaohao/Corpus/multimediate/sample_lists/val_next_speaker.csv", index_col=0)
valid_sample = valid_sample.reset_index(drop=True)
valid_record = sorted(list(set(valid_sample['recording'])))

gaze_df = pd.read_csv("./data/gaze_mean.csv", index_col=0)
gaze_df = gaze_df.rename(columns={'subject1': 'GazePos1', 'subject2': 'GazePos2', 'subject3': 'GazePos3','subject4': 'GazePos4'})
gaze_df = gaze_df.drop(columns=['start_time', 'end_time', 'label_1', 'label_2', 'label_3', 'label_4'])
train_gaze = gaze_df[gaze_df['recording'].isin(train_record)].reset_index(drop=True)
valid_gaze = gaze_df[gaze_df['recording'].isin(valid_record)].reset_index(drop=True)
train_gaze = train_gaze.drop(columns=['recording'])
valid_gaze = valid_gaze.drop(columns=['recording'])

talk_df = pd.read_csv("./data/talk_tend.csv", index_col=0)
talk_df = talk_df.drop(columns=['label_1', 'label_2', 'label_3', 'label_4'])
talk_df = talk_df.rename(columns={'subjectPos1': 'talkPos1', 'subjectPos2': 'talkPos2', 'subjectPos3': 'talkPos3','subjectPos4': 'talkPos4'})
train_talk = talk_df[talk_df['recording'].isin(train_record)].reset_index(drop=True)
valid_talk = talk_df[talk_df['recording'].isin(valid_record)].reset_index(drop=True)
train_talk = train_talk.drop(columns=['recording'])
valid_talk = valid_talk.drop(columns=['recording'])

train_ege = joblib.load("../Data/multimediate_egemap_train.pkl")
train_ege = train_ege.drop(columns=['recording','index','start_time', 'end_time', 'label_1', 'label_2', 'label_3', 'label_4'])
valid_ege = joblib.load("../Data/multimediate_egemap_valid.pkl")
valid_ege = valid_ege.drop(columns=['recording','index','start_time', 'end_time', 'label_1', 'label_2', 'label_3', 'label_4'])
ege_col = list(train_ege.columns)

# head pose 3D, gaze 3D, lips, jaw
openface_col = ['head_pose_x', 'head_pose_y', 'head_pose_z', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 
                'gaze_1_x', 'gaze_1_y', 'gaze_1_z','gaze_angle_x', 'gaze_angle_y', 'lip_r', 'lip_c', 'jaw_r', 'jaw_c']
train_openface = joblib.load("../Data/multimediate_OpenfaceStatic_train.pkl")
train_openface = train_openface.drop(columns=['index','start_time', 'end_time', 'label_1', 'label_2', 'label_3', 'label_4'])
train_openface = train_openface.assign(**pd.DataFrame(train_openface['subjectPos1'].values.tolist()).add_prefix('Pos1_'))
train_openface = train_openface.assign(**pd.DataFrame(train_openface['subjectPos2'].values.tolist()).add_prefix('Pos2_'))
train_openface = train_openface.assign(**pd.DataFrame(train_openface['subjectPos3'].values.tolist()).add_prefix('Pos3_'))
train_openface = train_openface.assign(**pd.DataFrame(train_openface['subjectPos4'].values.tolist()).add_prefix('Pos4_'))
train_openface = train_openface.drop(columns=['recording','subjectPos1', 'subjectPos2', 'subjectPos3', 'subjectPos4'])

valid_openface = joblib.load("../Data/multimediate_OpenfaceStatic_valid.pkl")
valid_openface = valid_openface.drop(columns=['index','start_time', 'end_time', 'label_1', 'label_2', 'label_3', 'label_4'])
valid_openface = valid_openface.assign(**pd.DataFrame(valid_openface['subjectPos1'].values.tolist()).add_prefix('Pos1_'))
valid_openface = valid_openface.assign(**pd.DataFrame(valid_openface['subjectPos2'].values.tolist()).add_prefix('Pos2_'))
valid_openface = valid_openface.assign(**pd.DataFrame(valid_openface['subjectPos3'].values.tolist()).add_prefix('Pos3_'))
valid_openface = valid_openface.assign(**pd.DataFrame(valid_openface['subjectPos4'].values.tolist()).add_prefix('Pos4_'))
valid_openface = valid_openface.drop(columns=['recording','subjectPos1', 'subjectPos2', 'subjectPos3', 'subjectPos4'])

# train = pd.concat([train_sample, train_gaze, train_talk, train_ege, train_openface], axis=1)
# valid = pd.concat([valid_sample, valid_gaze, valid_talk, valid_ege, valid_openface], axis=1)

train = pd.concat([train_sample, train_gaze, train_talk], axis=1)
valid = pd.concat([valid_sample, valid_gaze, valid_talk], axis=1)
overall_col = ['recording','label', 'Gaze', 'Talk']
fea = ['Gaze', 'Talk']

if ADD_EGE:
    overall_col = overall_col+ege_col
    fea = fea+ege_col
    train = pd.concat([train, train_ege], axis=1)
    valid = pd.concat([valid, valid_ege], axis=1)

if ADD_OPENFACE:
    overall_col = overall_col+openface_col
    fea = fea+openface_col
    train = pd.concat([train, train_openface], axis=1)
    valid = pd.concat([valid, valid_openface], axis=1)

train_input = pd.DataFrame(columns=overall_col)
valid_input = pd.DataFrame(columns=overall_col)

for i in range(1,5):
    temp = train.copy()
    temp_val = valid.copy()
    temp_op = ['Pos{}_{}'.format(i, j) for j in range(15)]
    
    temp_col = ['recording','label_{}'.format(i), 'GazePos{}'.format(i), 'talkPos{}'.format(i)]
    if ADD_EGE:
        temp_col = temp_col+ege_col
    if ADD_OPENFACE:
        temp_col = temp_col+temp_op
        
    temp = temp[temp_col]
    temp = temp.set_axis(overall_col, axis=1, inplace=False)
    
    temp_val = temp_val[temp_col]
    temp_val = temp_val.set_axis(overall_col, axis=1, inplace=False)
    
    train_input = pd.concat([train_input, temp], axis=0)
    valid_input = pd.concat([valid_input, temp_val], axis=0)

#%%
# import pdb;pdb.set_trace()
clf = tree.DecisionTreeClassifier(max_depth=4)

train_x = train_input[fea].values
train_y = train_input['label'].values.astype(int)

test_x = valid_input[fea].values
test_y = valid_input['label'].values.astype(int)

clf = clf.fit(train_x, train_y)

# tree.plot_tree(clf)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=fea, 
                                class_names=['silence', 'talk'], filled=True, rounded=True,
                                special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("iris")  

predict = clf.predict(test_x)

f1score = f1_score(test_y,predict, average='macro')
acc=accuracy_score(test_y,predict)
precision = precision_score(test_y, predict, average='macro')
recall = recall_score(test_y, predict, average='macro')
uar = (precision+recall)/2.0

print('f1 score: {:.4f}, acc: {:.4f}, recall: {:.4f}, UAR: {:.4f}'.format(precision, acc, recall, uar))




