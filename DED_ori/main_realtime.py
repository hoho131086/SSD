#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:33:48 2022

@author: shaohao
"""

import beam_search as bs
import argparse
import utils
import os
import sys
import numpy as np
import joblib
import logging
import json
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
import pdb
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(inputs, mode, method, args):
    """
    dialogs: dict, dialogs of the dataset.
    emo_dict: dict, emotions of all utterances.
    out_dict: dict, output logits of emotion classifier.
    """
    
    if mode == 'train':
        dialogs_train = joblib.load(inputs['dialog'])
        emo_dict_train = joblib.load(inputs['talk'])
        out_dict_train = joblib.load(inputs['prob'])
    
        dialogs = dialogs_train.copy()
        emo_dict = emo_dict_train.copy()
        out_dict = out_dict_train.copy()
    
    elif mode == 'test':
        dialogs_test = joblib.load(inputs['dialog'])
        emo_dict_test = joblib.load(inputs['talk'])
        out_dict_test = joblib.load(inputs['prob'])
    
        dialogs = dialogs_test.copy()
        emo_dict = emo_dict_test.copy()
        out_dict = out_dict_test.copy()
    
    elif mode == 'all':
        dialogs_train = joblib.load(inputs['dialog'])
        emo_dict_train = joblib.load(inputs['talk'])
        out_dict_train = joblib.load(inputs['prob'])
    
        dialogs = dialogs_train.copy()
        emo_dict = emo_dict_train.copy()
        out_dict = out_dict_train.copy()
        
        dialogs_test = joblib.load(inputs['dialog'])
        emo_dict_test = joblib.load(inputs['talk'])
        out_dict_test = joblib.load(inputs['prob'])
        
        dialogs.update(dialogs_test)
        emo_dict.update(emo_dict_test)
        out_dict.update(out_dict_test)
    
    # pdb.set_trace()
    recording_lst = sorted(set([i.split('_')[0] for i in dialogs.keys()]))
    performance = pd.DataFrame(columns=['recording', 'subjectPos1', 'subjectPos2', 'subjectPos3', 'subjectPos4'])
    performance['recording']=recording_lst
    performance.index = performance['recording']
    
    # set log
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO,
                        datefmt='%I:%M:%S')
    
    # Split dialogs by speakers
    # spk_dialogs = utils.split_dialog(dialogs)
    spk_dialogs = dialogs
    logging.info("Average number of speaker's utterances per dialog: %.3f" % np.mean(
                                                      [len(i) for i in spk_dialogs.values()]))
    
    
    # arguments
    # args = parse_args()
    
    print('=' * 60 + '\n')
    logging.info('Parameters are:\n%s\n', json.dumps(vars(args), sort_keys=False, indent=4))
    print('=' * 60 + '\n')
    # pdb.set_trace()
    
    if args.transition_bias > 0:
        # Use given p_0
        p_0 = args.transition_bias
    
    else:
        # Estimate p_0 of ALL dialogs.
        p_0, total_transit = utils.transition_bias(spk_dialogs, emo_dict)
        print("\n"+"#"*50)
        logging.info('p_0: %.3f , total transition: %d\n' % (p_0, total_transit))
        print("#"*50)
        session = list(spk_dialogs.keys())
        bias_dict = utils.get_val_bias(spk_dialogs, emo_dict, session)
        # pdb.set_trace()  
        print("#"*50+"\n")
    
    # pdb.set_trace()
    trace = []
    label = []
    org_pred = []
    DED = bs.BeamSearch_RealTime(p_0, args.crp_alpha, args.num_state, 
                                args.beam_size, args.test_iteration, emo_dict, out_dict, 'no')
    
    indiv_output = {}
    for i, dia in enumerate(dialogs):
        logging.info("Decoding dialog: {}/{}, {}".format(i,len(dialogs),dia))
      
        if args.transition_bias < 0:
            DED.transition_bias = bias_dict[dia[:13]] 
        
        # Beam search decoder
        beam = [bs.BeamState([], [], 0, [0 for i in range(args.num_state)], []) for _ in range(args.beam_size)]
        output = []
        for t in range(len(dialogs[dia])):
            # pdb.set_trace()
            utt_id = dialogs[dia][t]
            beam_new, out = DED.decode(t, beam, utt_id) 
            beam = beam_new.copy()
                
            if method == 'me_other':
                step_output = [1 if i == 0 else 0 for i in out]
                output += step_output
                
            elif method == 'indiv':
                output += out

        trace += output
        if method == 'me_other':
            label += [utils.convert_to_index_me_other_label(emo_dict[utt]) for utt in dialogs[dia]]
            temp_org = [np.argmax(out_dict[utt]) for utt in dialogs[dia]]
            temp_org = [1 if i == 0 else 0 for i in temp_org]
            org_pred += temp_org
            
        elif method == 'indiv':
            label += [utils.convert_to_index_individual_label(emo_dict[utt]) for utt in dialogs[dia]]
            org_pred += [np.argmax(out_dict[utt]) for utt in dialogs[dia]]
        # pdb.set_trace()
        
        if args.verbosity > 0:
            logging.info("Output: {}\n".format(out))
        
        indiv_output[dia+'_pred'] = output
        indiv_output[dia+'_gt'] = [utils.convert_to_index_individual_label(emo_dict[utt]) for utt in dialogs[dia]]         
        indiv_output[dia+'_org'] = [np.argmax(out_dict[utt]) for utt in dialogs[dia]]   
        
        # temp_label = [utils.convert_to_index_me_other_label(emo_dict[utt]) for utt in dialogs[dia]]
        # uar = recall_score(temp_label, output, average='macro')
        # precision = precision_score(temp_label, output, average = 'macro')
        # f1 = f1_score(temp_label, output, average='macro')
        # performance.loc[dia.split('_')[0], 'subjectPos{}'.format(dia.split('_')[-1])] = uar
        
        # pdb.set_trace()
    
    # performance.to_csv('me_other_performance.csv')
    print("#"*50+"\n")
    # Print the results of emotino classifier module
    acc, uar, precision, f1, conf = utils.evaluate(org_pred, label)
    logging.info('Original performance: uar: %.4f, acc: %.4f, f1: %.4f, precision: %.4f' % (uar, acc, f1, precision))
    
    # Eval ded outputs
    # pdb.set_trace()
    acc, uar, precision, f1, conf = utils.evaluate(trace, label)
    logging.info('DED %s realtime performance: uar: %.4f, acc: %.4f, f1: %.4f, precision: %.4f' % (method, uar, acc, f1, precision))
    logging.info('Confusion matrix:\n%s' % conf)
    return indiv_output    

if __name__ == '__main__':  
    
    parser = argparse.ArgumentParser(
        description='A tensorflow implementation of end-to-end speech recognition system:'
                    'Listen, Attend and Spell (LAS)')    
    # feature arguments
    parser.add_argument('--beam_size', 
                        type=int, 
                        default=5, 
                        help='Beam size.')
    parser.add_argument('--crp_alpha', 
                        type=int, 
                        default=1, 
                        help='Alpha in chinese resturant process. See eq (6).')
    # origin 0.445
    # me yes no 0.153
    # me other 0.248
    parser.add_argument('--transition_bias', 
                        type=int, 
                        default=0.705, 
                        help='Transition bias, if not given, p_0 will be estimated from'
                             'training data. See eq (5)')
    parser.add_argument('--num_state', 
                        type=int, 
                        default=5, 
                        help='Number of emotion states. Note that the states are bounded.')
    parser.add_argument('--test_iteration', 
                        type=int, 
                        default=1, 
                        help='Before decoding, we duplicate a dialog K times and concatenate'
                             'them together. After decoding, We return the prediction of last' 
                             'duplicated sequence.' 
                             'See: https://github.com/google/uis-rnn/blob/master/uisrnn/arguments.py')
    parser.add_argument('--verbosity', 
                        type=int, 
                        default=0, 
                        help='Set the verbosity.')
    parser.add_argument('--out_dir', 
                        type=str, 
                        default='./outputs', 
                        help='Output directory.')
    parser.add_argument('--result_file', 
                        type=str, 
                        default='results', 
                        help='Save results.')
    args = parser.parse_args()
    
    # mode = 'test'
    # method = 'indiv'
    # fea = 'talk_me_other/cv4'
    # inputs = {'talk':'data/{}/{}_{}_talk.pkl'.format(fea,method, mode), 
    #           'prob':'data/{}/{}_{}_prob.pkl'.format(fea,method, mode), 
    #           'dialog':'data/{}/{}_{}_dialog.pkl'.format(fea,method, mode)}
    
    mode = 'test'
    method = 'indiv'
    fea = 'indiv'
    inputs = {'talk':'data/{}/GATE_4/{}_{}_talk.pkl'.format(fea,method, mode), 
              'prob':'data/{}/GATE_4/{}_{}_prob.pkl'.format(fea,method, mode), 
              'dialog':'data/{}/GATE_4/{}_{}_dialog.pkl'.format(fea,method, mode)}
    
    # if mode == 'test' or mode == 'all':
    #     if method == 'me_other':
    #         args.transition_bias = 0.7
    #     elif method == 'indiv':
    #         args.transition_bias = 0.7
    # else:
    #     args.transition_bias = -1
    
    if method == 'me_other':
        args.num_state = 3
    elif method == 'indiv':
        args.num_state = 2
    
    print(fea)
    out = main(inputs, mode, method, args)
    # joblib.dump(out, './real_time_result.pkl')

    
  
  
  
  
