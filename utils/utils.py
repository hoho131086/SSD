#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:45:06 2022

@author: shaohao
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import tqdm
import pdb
import pickle


def get_data_y(index, df, lst):

    clear = df[df['group'].isin(lst)]
    Data_1 = np.vstack(clear['input_1'])
    Data_2 = np.vstack(clear['input_2'])
    Data_3 = np.vstack(clear['input_3'])
    if 'input_4' in list(clear.columns):
        Data_4 = np.vstack(clear['input_4'])
        
    y_1 = np.vstack(clear['nextspeaker_1'])
    y_2 = np.vstack(clear['nextspeaker_2'])
    y_3 = np.vstack(clear['nextspeaker_3'])
    if 'nextspeaker_4' in list(clear.columns):
        y_4 = np.vstack(clear['nextspeaker_4'])
        return Data_1.astype(np.float), Data_2.astype(np.float), Data_3.astype(np.float), Data_4.astype(np.float),\
            y_1.ravel(), y_2.ravel(), y_3.ravel(), y_4.ravel()
    else:
        return Data_1.astype(np.float), Data_2.astype(np.float), Data_3.astype(np.float),\
            y_1.ravel(), y_2.ravel(), y_3.ravel()

def pad_or_truncate(some_list, target_len):
    return ['0']*(target_len - len(some_list)) + some_list[-target_len:]

def ipa_encode(df, mode):
    
    if mode == 1:
        output = df['ipa_seq']
        output = pad_or_truncate(output, 1)
        output = [int(i.split('_')[-1]) for i in output]
        return output
    
    elif mode == 2:
        output = df['ipa_seq']
        output = pad_or_truncate(output, 2)
        output = [int(i.split('_')[-1]) for i in output]
        return output
    
    elif mode ==3:
        output = df['ipa_seq']
        output = pad_or_truncate(output, 2)
        output = [int(i.split('_')[-1]) for i in output]
        
        output = output + [df['person_control_ratio_1'], 
                           df['person_control_ratio_2'], 
                           df['person_control_ratio_3'], 
                           df['conv_duration_control_ratio_1'],
                           df['conv_duration_control_ratio_2'],
                           df['conv_duration_control_ratio_3'],
                           df['duration_control_ratio_1'],
                           df['duration_control_ratio_2'],
                           df['duration_control_ratio_3']]
        return output
    
    elif mode == 4:
        output = df['ipa_seq']
        output = pad_or_truncate(output, 2)
        output = [int(i.split('_')[-1]) for i in output]
        
        output_1 = output + [df['person_control_ratio_1'], 
                           df['conv_duration_control_ratio_1'],
                           df['duration_control_ratio_1']]
        
        output_2 = output + [df['person_control_ratio_2'], 
                           df['conv_duration_control_ratio_2'],
                           df['duration_control_ratio_2']]
        
        output_3 = output + [df['person_control_ratio_3'], 
                           df['conv_duration_control_ratio_3'],
                           df['duration_control_ratio_3']]
        return pd.Series([output_1, output_2, output_3], index=['input_1', 'input_2', 'input_3'])

#%%
def get_data(args):
    
    CORPUS = args.corpus
    FEATURE = args.feature
    IPA_MODE = args.mode
    
    if CORPUS == 'ntuba':
        if FEATURE == 'ipa':
            data = joblib.load('../Data/ntuba_ipa.pkl')
            if IPA_MODE == 'baseline1':
                data['input_1'] = data.apply(ipa_encode, mode=1, axis=1)
                data['input_2'] = data['input_1']
                data['input_3'] = data['input_1']
                
            elif IPA_MODE == 'baseline2':
                data['input_1'] = data.apply(ipa_encode, mode=2, axis=1)
                data['input_2'] = data['input_1']
                data['input_3'] = data['input_1']
                
            elif IPA_MODE == 'summary':
                if args.group_feature == 'yes':
                    data['input_1'] = data.apply(lambda x: [x['person_control_ratio_1'], 
                                                          x['person_control_ratio_2'], 
                                                          x['person_control_ratio_3'], 
                                                          x['conv_duration_control_ratio_1'],
                                                          x['conv_duration_control_ratio_2'],
                                                          x['conv_duration_control_ratio_3'],
                                                          x['duration_control_ratio_1'],
                                                          x['duration_control_ratio_2'],
                                                          x['duration_control_ratio_3']],axis=1)
                    
                    data['input_2'] = data['input_1']
                    data['input_3'] = data['input_1']
                    
                else:
                    data['input_1'] = data.apply(lambda x: [x['person_control_ratio_1'], 
                                                            x['conv_duration_control_ratio_1'],
                                                            x['duration_control_ratio_1']],axis=1)
                    data['input_2'] = data.apply(lambda x: [x['person_control_ratio_2'], 
                                                            x['conv_duration_control_ratio_2'],
                                                            x['duration_control_ratio_2']],axis=1)
                    data['input_3'] = data.apply(lambda x: [x['person_control_ratio_3'], 
                                                            x['conv_duration_control_ratio_3'],
                                                            x['duration_control_ratio_3']],axis=1)
                    
                
            elif IPA_MODE == 'full':
                if args.group_feature == 'yes':
                    data['input_1'] = data.apply(ipa_encode, mode=3, axis=1)
                    data['input_2'] = data['input_1']
                    data['input_3'] = data['input_1']
                else:
                    data[['input_1','input_2','input_3']] = data.apply(ipa_encode, mode=4, axis=1)
            
        elif FEATURE == 'egemap':
            data = joblib.load('../Data/ntuba_multi_egemap.pkl')
            EGE_COL = list(data.columns)[-88:]
            data['input_1'] = data.apply(lambda x: x[EGE_COL].values, axis=1)
            data['input_2'] =  data['input_1']
            data['input_3'] =  data['input_1']
            data = data.drop(columns=EGE_COL)
            data = data.rename(columns={'label_1': 'nextspeaker_1','label_2': 'nextspeaker_2', 
                                        'label_3': 'nextspeaker_3'})
            
        elif FEATURE == 'OpenfaceStatic':
            data = joblib.load('../Data/ntuba_OpenfaceStatic.pkl')
            OPEN_COL = list(data.columns)[-3:]
            if args.group_feature == 'yes':
                data['input_1'] = data.apply(lambda x: np.hstack(x[OPEN_COL]), axis=1)
                data['input_2'] =  data['input_1']
                data['input_3'] =  data['input_1']

            else:
                data['input_1'] = data.apply(lambda x: np.hstack(x[OPEN_COL[0]]), axis=1)
                data['input_2'] = data.apply(lambda x: np.hstack(x[OPEN_COL[1]]), axis=1)
                data['input_3'] = data.apply(lambda x: np.hstack(x[OPEN_COL[2]]), axis=1)
                
            data = data.drop(columns=OPEN_COL)
            data = data.rename(columns={'label_1': 'nextspeaker_1','label_2': 'nextspeaker_2', 
                                        'label_3': 'nextspeaker_3'})
        
        elif FEATURE == 'all':
            
            if args.group_feature == 'yes':
                data = joblib.load('../Data/ntuba_group_all.pkl')
                # data = joblib.load('../Data/ntuba_group_no_ipa.pkl')
                
            else:
                data = joblib.load('../Data/ntuba_indiv_all.pkl')
                # data = joblib.load('../Data/ntuba_indiv_no_ipa.pkl')
        
        
        else:
            print("no this features")
            pdb.set_trace()
        
        unique = data.drop_duplicates(subset=['group'])
        return data, unique
    
######################################################
    elif CORPUS == 'multimediate':
        
        if FEATURE == 'egemap':
            train_data = joblib.load("../Data/multimediate_egemap_train.pkl")
            val_data = joblib.load("../Data/multimediate_egemap_valid.pkl")
            FEA_COL = list(train_data.columns)[-88:]
            train_data['input_1'] = train_data.apply(lambda x: x[FEA_COL].values, axis=1)
            train_data['input_2'] = train_data['input_1']
            train_data['input_3'] = train_data['input_1']
            train_data['input_4'] = train_data['input_1']
            train_data = train_data.drop(columns=FEA_COL)
            train_data = train_data.rename(columns={'recording': 'group', 'label_1': 'nextspeaker_1', 
                                                    'label_2': 'nextspeaker_2', 'label_3': 'nextspeaker_3', 
                                                    'label_4': 'nextspeaker_4'})
            
            val_data['input_1'] = val_data.apply(lambda x: x[FEA_COL].values, axis=1)
            val_data['input_2'] = val_data['input_1']
            val_data['input_3'] = val_data['input_1']
            val_data['input_4'] = val_data['input_1']
            
            val_data = val_data.drop(columns=FEA_COL)
            val_data = val_data.rename(columns={'recording': 'group', 'label_1': 'nextspeaker_1', 
                                                    'label_2': 'nextspeaker_2', 'label_3': 'nextspeaker_3', 
                                                    'label_4': 'nextspeaker_4'})
            
        elif 'Openface' in FEATURE:
            train_data = joblib.load("../Data/multimediate_OpenfaceStatic_train.pkl")
            val_data = joblib.load("../Data/multimediate_OpenfaceStatic_valid.pkl")
            FEA_COL = list(train_data.columns)[-4:]
            if args.group_feature == 'yes':
                train_data['input_1'] = train_data.apply(lambda x: np.hstack(x[FEA_COL]), axis=1)
                train_data['input_2'] = train_data['input_1']
                train_data['input_3'] = train_data['input_1']
                train_data['input_4'] = train_data['input_1']
                
                val_data['input_1'] = val_data.apply(lambda x: np.hstack(x[FEA_COL]), axis=1)
                val_data['input_2'] = val_data['input_1']
                val_data['input_3'] = val_data['input_1']
                val_data['input_4'] = val_data['input_1']
                
            else:
                train_data['input_1'] = train_data.apply(lambda x: np.hstack(x[FEA_COL[0]]), axis=1)
                train_data['input_2'] = train_data.apply(lambda x: np.hstack(x[FEA_COL[1]]), axis=1)
                train_data['input_3'] = train_data.apply(lambda x: np.hstack(x[FEA_COL[2]]), axis=1)
                train_data['input_4'] = train_data.apply(lambda x: np.hstack(x[FEA_COL[3]]), axis=1)
                
                val_data['input_1'] = val_data.apply(lambda x: np.hstack(x[FEA_COL[0]]), axis=1)
                val_data['input_2'] = val_data.apply(lambda x: np.hstack(x[FEA_COL[1]]), axis=1)
                val_data['input_3'] = val_data.apply(lambda x: np.hstack(x[FEA_COL[2]]), axis=1)
                val_data['input_4'] = val_data.apply(lambda x: np.hstack(x[FEA_COL[3]]), axis=1)
                
            train_data = train_data.drop(columns=FEA_COL)
            train_data = train_data.rename(columns={'recording': 'group', 'label_1': 'nextspeaker_1', 
                                                    'label_2': 'nextspeaker_2', 'label_3': 'nextspeaker_3', 
                                                    'label_4': 'nextspeaker_4'})
            
            val_data = val_data.drop(columns=FEA_COL)
            val_data = val_data.rename(columns={'recording': 'group', 'label_1': 'nextspeaker_1', 
                                                    'label_2': 'nextspeaker_2', 'label_3': 'nextspeaker_3', 
                                                    'label_4': 'nextspeaker_4'})
        
        elif FEATURE == 'all':
            if args.group_feature == 'yes':
                train_data = joblib.load("../Data/multimediate_group_all_train.pkl")
                val_data = joblib.load("../Data/multimediate_group_all_valid.pkl")
            else:
                train_data = joblib.load("../Data/multimediate_indiv_all_train.pkl")
                val_data = joblib.load("../Data/multimediate_indiv_all_valid.pkl")
            
        else:
            print("no this features")
            pdb.set_trace()
        
        unique = train_data.drop_duplicates(subset=['group'])
        Test_1 = np.vstack(val_data['input_1']).astype(np.float)
        Test_2 = np.vstack(val_data['input_2']).astype(np.float)
        Test_3 = np.vstack(val_data['input_3']).astype(np.float)
        Test_4 = np.vstack(val_data['input_4']).astype(np.float)
        Test_y_1 = np.vstack(val_data['nextspeaker_1']).ravel()
        Test_y_2 = np.vstack(val_data['nextspeaker_2']).ravel()
        Test_y_3 = np.vstack(val_data['nextspeaker_3']).ravel()
        Test_y_4 = np.vstack(val_data['nextspeaker_4']).ravel()
        
        return train_data, Test_1, Test_2, Test_3, Test_4, Test_y_1, Test_y_2, Test_y_3, Test_y_4, unique

#%%   
def run_eval_ntuba(seed, unique, data, args):    
    
    np.random.seed(seed)
    random.seed(seed)
    ground_truth = []
    pred = []
    
    FEATURE = args.feature
    MODEL = args.model
    CORPUS = 'ntuba'
    
    CV_FOLD = 5
    skf = KFold(n_splits=CV_FOLD)
    
    for fold, (train_index, test_index) in tqdm.tqdm(enumerate(skf.split(unique))):
        ground_truth = []
        pred = []
        
        origin_uar = -1
        
        valid_index=sorted(random.sample(train_index.tolist(),len(test_index)))
        train_index=[i for i in train_index if i not in valid_index]
        
        train_lst = list(unique.iloc[train_index]['group'])
        val_lst = list(unique.iloc[valid_index]['group'])
        test_lst = list(unique.iloc[test_index]['group'])
        
        if args.model=='randomforest':
            model_1 = RandomForestClassifier()
            model_2 = RandomForestClassifier()
            model_3 = RandomForestClassifier()
        elif args.model=='svm':
            model_1 = SVC(gamma='auto')
            model_2 = SVC(gamma='auto')
            model_3 = SVC(gamma='auto')
        
        Train_1, Train_2, Train_3 , Train_y_1, Train_y_2, Train_y_3 = get_data_y(train_index, data, train_lst)           
        Valid_1, Valid_2, Valid_3, Valid_y_1, Valid_y_2, Valid_y_3 = get_data_y(valid_index, data, val_lst)            
        Test_1, Test_2, Test_3, Test_y_1, Test_y_2, Test_y_3 = get_data_y(test_index, data, test_lst)
        
        temp_train = np.vstack([Train_1, Train_2, Train_3])
        
        # pdb.set_trace()
        sc = StandardScaler()
        sc.fit(temp_train)
        Train_1 = sc.transform(Train_1)
        Train_2 = sc.transform(Train_2)
        Train_3 = sc.transform(Train_3)
        
        Valid_1 = sc.transform(Valid_1)
        Valid_2 = sc.transform(Valid_2)
        Valid_3 = sc.transform(Valid_3)
        
        Test_1 = sc.transform(Test_1)
        Test_2 = sc.transform(Test_2)
        Test_3 = sc.transform(Test_3)
        
        
        model_1.fit(Train_1, Train_y_1)
        model_2.fit(Train_2, Train_y_2)
        model_3.fit(Train_3, Train_y_3)
        
        y_pred_1 = model_1.predict(Valid_1)
        y_pred_2 = model_2.predict(Valid_2)
        y_pred_3 = model_3.predict(Valid_3)
        
        pred.extend(list(y_pred_1))
        pred.extend(list(y_pred_2))
        pred.extend(list(y_pred_3))
        ground_truth.extend(list(Valid_y_1))
        ground_truth.extend(list(Valid_y_2))
        ground_truth.extend(list(Valid_y_3))
         
        f1score = f1_score(ground_truth,pred, average='macro', zero_division=1)
        acc=accuracy_score(ground_truth,pred)
        precision = precision_score(ground_truth, pred, average='macro', zero_division=1)
        recall = recall_score(ground_truth, pred, average='macro', zero_division=1)
        uar = (precision+recall)/2.0
        
        if uar >= origin_uar:
            origin_uar = uar
            filename_1 = '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/best_model/{}_{}_{}_group_{}_1.sav'.format(CORPUS, FEATURE, MODEL, args.group_feature)
            joblib.dump(model_1, filename_1)
            filename_2 = '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/best_model/{}_{}_{}_group_{}_2.sav'.format(CORPUS, FEATURE, MODEL, args.group_feature)
            joblib.dump(model_2, filename_2)
            filename_3 = '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/best_model/{}_{}_{}_group_{}_3.sav'.format(CORPUS, FEATURE, MODEL, args.group_feature)
            joblib.dump(model_3, filename_3)
    
    loaded_model_1 = joblib.load(filename_1)
    loaded_model_2 = joblib.load(filename_2)
    loaded_model_3 = joblib.load(filename_3)
    
    y_pred_1 = loaded_model_1.predict(Test_1)
    y_pred_2 = loaded_model_2.predict(Test_2)
    y_pred_3 = loaded_model_3.predict(Test_3)
    
    TEST_pred = []
    TEST_ground_truth = []
    
    TEST_pred.extend(list(y_pred_1))
    TEST_pred.extend(list(y_pred_2))
    TEST_pred.extend(list(y_pred_3))
    TEST_ground_truth.extend(list(Test_y_1))
    TEST_ground_truth.extend(list(Test_y_2))
    TEST_ground_truth.extend(list(Test_y_3))
        
    return TEST_pred, TEST_ground_truth


def run_eval_multi(seed, unique, train_data, Test_1, Test_2, Test_3, Test_4, Test_y_1, Test_y_2, Test_y_3, Test_y_4, args):
    
    CV_FOLD = 5
    skf = KFold(n_splits=CV_FOLD)
    
    FEATURE = args.feature
    MODEL = args.model
    CORPUS = 'multi'

    
    for fold, (train_index, val_index) in tqdm.tqdm(enumerate(skf.split(unique))):
        ground_truth = []
        pred = []
        origin_uar = -1
        
        train_lst = list(unique.iloc[train_index]['group'])
        val_lst = list(unique.iloc[val_index]['group'])
        
        Train_1, Train_2, Train_3, Train_4 , Train_y_1, Train_y_2, Train_y_3, Train_y_4 = get_data_y(train_index, train_data, train_lst)
        Val_1, Val_2, Val_3, Val_4, Val_y_1, Val_y_2, Val_y_3, Val_y_4 = get_data_y(val_index, train_data, val_lst)
        if MODEL=='randomforest':
            model_1 = RandomForestClassifier()
            model_2 = RandomForestClassifier()
            model_3 = RandomForestClassifier()
            model_4 = RandomForestClassifier()
        elif MODEL=='svm':
            model_1 = SVC(gamma='auto')
            model_2 = SVC(gamma='auto')
            model_3 = SVC(gamma='auto')
            model_4 = SVC(gamma='auto')
        
        temp_train = np.vstack([Train_1, Train_2, Train_3, Train_4])
        sc = StandardScaler()
        sc.fit(temp_train)
        Train_1 = sc.transform(Train_1)
        Train_2 = sc.transform(Train_2)
        Train_3 = sc.transform(Train_3)
        Train_4 = sc.transform(Train_4)
        
        Val_1 = sc.transform(Val_1)
        Val_2 = sc.transform(Val_2)
        Val_3 = sc.transform(Val_3)
        Val_4 = sc.transform(Val_4)
        
        model_1.fit(Train_1, Train_y_1)
        model_2.fit(Train_2, Train_y_2)
        model_3.fit(Train_3, Train_y_3)
        model_4.fit(Train_4, Train_y_4)
        y_pred_1 = model_1.predict(Val_1)
        y_pred_2 = model_2.predict(Val_2)
        y_pred_3 = model_3.predict(Val_3)
        y_pred_4 = model_4.predict(Val_4)
        
        pred.extend(list(y_pred_1))
        pred.extend(list(y_pred_2))
        pred.extend(list(y_pred_3))
        pred.extend(list(y_pred_4))
        ground_truth.extend(list(Val_y_1))
        ground_truth.extend(list(Val_y_2))
        ground_truth.extend(list(Val_y_3))
        ground_truth.extend(list(Val_y_4))
         
        f1score = f1_score(ground_truth,pred, average='macro', zero_division=1)
        acc=accuracy_score(ground_truth,pred)
        precision = precision_score(ground_truth, pred, average='macro', zero_division=1)
        recall = recall_score(ground_truth, pred, average='macro', zero_division=1)
        uar = (precision+recall)/2.0
        
        if uar >= origin_uar:
            origin_uar = uar
            filename_1 = '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/best_model/{}_{}_{}_group_{}_1.sav'.format(CORPUS, FEATURE, MODEL, args.group_feature)
            joblib.dump(model_1, filename_1)
            filename_2 = '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/best_model/{}_{}_{}_group_{}_2.sav'.format(CORPUS, FEATURE, MODEL, args.group_feature)
            joblib.dump(model_2, filename_2)
            filename_3 = '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/best_model/{}_{}_{}_group_{}_3.sav'.format(CORPUS, FEATURE, MODEL, args.group_feature)
            joblib.dump(model_3, filename_3)
            filename_4 = '/homes/GPU2/shaohao/turn_taking/turn_changing/next_speaker/best_model/{}_{}_{}_group_{}_4.sav'.format(CORPUS, FEATURE, MODEL, args.group_feature)
            joblib.dump(model_4, filename_4)
            
            Test_1 = sc.transform(Test_1)
            Test_2 = sc.transform(Test_2)
            Test_3 = sc.transform(Test_3)
            Test_4 = sc.transform(Test_4)
    
    loaded_model_1 = joblib.load(filename_1)
    loaded_model_2 = joblib.load(filename_2)
    loaded_model_3 = joblib.load(filename_3)
    loaded_model_4 = joblib.load(filename_4)
    
    y_pred_1 = loaded_model_1.predict(Test_1)
    y_pred_2 = loaded_model_2.predict(Test_2)
    y_pred_3 = loaded_model_3.predict(Test_3)
    y_pred_4 = loaded_model_4.predict(Test_4)
    
    TEST_pred = []
    TEST_ground_truth = []
    
    TEST_pred.extend(list(y_pred_1))
    TEST_pred.extend(list(y_pred_2))
    TEST_pred.extend(list(y_pred_3))
    TEST_pred.extend(list(y_pred_4))
    TEST_ground_truth.extend(list(Test_y_1))
    TEST_ground_truth.extend(list(Test_y_2))
    TEST_ground_truth.extend(list(Test_y_3))
    TEST_ground_truth.extend(list(Test_y_4))
        
    return TEST_pred, TEST_ground_truth
    
    
    
    
    
    
    



