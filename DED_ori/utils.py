#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:44:37 2022

@author: shaohao
"""

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
import pdb

def split_dialog(dialogs):
    """Split utterances in a dialog into a set of speaker's utternaces in that dialog.
       See eq (5) in the paper.
    Arg:
      dialogs: dict, for example, utterances of two speakers in dialog_01: 
              {dialog_01: [utt_spk01_1, utt_spk02_1, utt_spk01_2, ...]}.
    Return:
      spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
              {dialog_01_spk01: [utt_spk01_1, utt_spk01_2, ...],
               dialog_01_spk02: [utt_spk02_1, utt_spk02_2, ...]}
    """
    # pdb.set_trace()
    spk_dialogs = {}
    for dialog_id in dialogs.keys():
        spk_dialogs[dialog_id+'_M'] = []
        spk_dialogs[dialog_id+'_F'] = []
        for utt_id in dialogs[dialog_id]:
            if utt_id[-4] == 'M':
                spk_dialogs[dialog_id+'_M'].append(utt_id)
            elif utt_id[-4] == 'F':
                spk_dialogs[dialog_id+'_F'].append(utt_id)
    
    return spk_dialogs

def transition_bias(spk_dialogs, emo, val=None):
    """Estimate the transition bias of emotion. See eq (5) in the paper.
    Args:
      spk_dialogs: dict, a collection of speakers' utterances in dialogs. for example:
      emo: dict, map from utt_id to emotion state.
      val: str, validation session. If given, calculate the trainsition bias except 
           dialogs in the validation session. For example, 'Ses01'.
    
    Returns: 
      bias: p_0 in eq (4).
    """
    # pdb.set_trace()
    transit_num = 0
    total_transit = 0
    count = 0
    num = 0
    for dialog_id in spk_dialogs.values():
        if val and val == dialog_id[0][:11]:
            continue
    
        for entry in range(len(dialog_id) - 1):
            # if len(set(emo[dialog_id[entry]])&set(emo[dialog_id[entry + 1]]))==0:
            #     transit_num += 1
            transit_num += (emo[dialog_id[entry]] != emo[dialog_id[entry + 1]])
            # if emo[dialog_id[entry]]==emo[dialog_id[entry+1]]:
            #     continue
            # transit_num -= (len(emo[dialog_id[entry]]+emo[dialog_id[entry + 1]]) / \
            #                 len((set(emo[dialog_id[entry]]+emo[dialog_id[entry + 1]])))-2)
        # pdb.set_trace()
             
        total_transit += (len(dialog_id) - 1)
    
    bias = (transit_num + 1) / total_transit
    
    return bias, total_transit

def get_val_bias(dialog, emo_dict, session):
    """Get p_0 estimated from training sessions."""

    # session = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    bias_dict = {}
    for i in range(len(session)):
        val = session[i]
        train_sessions = session[:i] + session[i+1:]
        p_0, _ = transition_bias(dialog, emo_dict, val)
        # print("Transition bias of { %s }: %.3f" % (' ,'.join(train_sessions), p_0))
        bias_dict[val] = p_0

    return bias_dict

def find_last_idx(trace_speakers, speaker):
    """Find the index of speaker's last utterance."""
    for i in range(len(trace_speakers)):
        if trace_speakers[len(trace_speakers) - (i+1)] == speaker:
            return len(trace_speakers) - (i+1)

def cross_entropy(targets, predictions, epsilon=1e-12):
    """Computes cross entropy between targets (one-hot) and predictions. 
    Args: 
      targets: (1, num_state) ndarray.   
      predictions: (1, num_state) ndarray.
    
    Returns: 
      cross entropy loss.
    """
    targets = np.array(targets)
    predictions = np.array(predictions)
    ce = -np.sum(targets*predictions)
    return ce

def convert_to_index(emotion):
    """convert emotion to index """
    map_emo = {'most_speaker':0, 'second_speaker':1, 'third_speaker':2, 'least_speaker':3}
    out = [0,0,0,0]
    for i in emotion:
        out[map_emo[i]]=1
    return out

def convert_to_index_memo(emotion, memo, thresh):
    """convert emotion to index """
    out = []
    for ind, item in enumerate(emotion):
        temp = [0,0,0,0]
        # temp[item] = 1
        
        temp_dict = memo[ind]
        # temp[min(d, key=d.get)]

        temp_index = [k for k,v in temp_dict.items() if v < thresh]
        # pdb.set_trace()
        for i in temp_index:
            temp[i] = 1
        # pdb.set_trace()
        out.append(temp)
    return out

def convert_to_index_candidate(emotion, thresh):
    """convert emotion to index """
    out = []
    for ind in range(len(emotion[0])):
        temp = [0,0,0,0]
        
        for i in range(len(emotion)):
            temp[emotion[i][ind]]+=1
        
        temp = [int(i>thresh) for i in temp]
        
        # pdb.set_trace()
        out.append(temp)
    return out

def convert_to_index_out(emotion):
    """convert emotion to index """
    out = []
    for ind, item in enumerate(emotion):
        temp = [0,0,0,0]
        temp[item] = 1
        
        out.append(temp)
    return out

def convert_to_index_individual_label(emotion):
    """convert emotion to index """
    if emotion == 'talk':
        return 1
    else:
        return 0

def convert_to_index_me_other_label(emotion):
    """convert emotion to index """
    if emotion == 'me':
        return 1
    else:
        return 0


def evaluate(trace, label):
    # Only evaluate utterances labeled in defined 4 emotion states
    label, trace = np.array(label).reshape(-1,1), np.array(trace).reshape(-1,1)
    # index = [label != -1]
    # label, trace = label[index], trace[index]
    uar = recall_score(label, trace, average='macro')
    acc = accuracy_score(label, trace)
    precision = precision_score(label, trace, average = 'macro')
    f1 = f1_score(label, trace, average = 'macro')
    # uar = (recall+precision)/2.0
    
    return acc, uar, precision, f1, confusion_matrix(label, trace)


# if __name__ == '__main__':
#     dialog = {'Ses05M_script03_2_M': ['Ses05M_script03_2_M042', 'Ses05M_script03_2_M043', 
#                 'Ses05M_script03_2_M044', 'Ses05M_script03_2_M045']}
#     emo = {'Ses05M_script03_2_M042': 'ang', 'Ses05M_script03_2_M043': 'ang', 
#                 'Ses05M_script03_2_M044': 'ang', 'Ses05M_script03_2_M045': 'ang'}

#     spk_dialog = split_dialog(dialog)
#     bias, total_transit = transition_bias(spk_dialog, emo)
#     crp_alpha = 1
#     print('Transition bias: {} , Total transition: {}'.format(bias, total_transit))

