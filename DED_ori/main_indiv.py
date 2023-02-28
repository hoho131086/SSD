#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:58:21 2022

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
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, classification_report
import pdb
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(mode, fea, args):
    """
    dialogs: dict, dialogs of the dataset.
    emo_dict: dict, emotions of all utterances.
    out_dict: dict, output logits of emotion classifier.
    """
    if mode == 'train':
        dialogs_train = joblib.load('data/{}/indiv_train_dialog.pkl'.format(fea))
        emo_dict_train = joblib.load('data/{}/indiv_train_talk.pkl'.format(fea))
        out_dict_train = joblib.load('data/{}/indiv_train_prob.pkl'.format(fea))   
        
        dialogs = dialogs_train.copy()
        emo_dict = emo_dict_train.copy()
        out_dict = out_dict_train.copy()
    
    elif mode == 'test':
        dialogs_test = joblib.load('data/{}/indiv_test_dialog.pkl'.format(fea))
        emo_dict_test = joblib.load('data/{}/indiv_test_talk.pkl'.format(fea))
        out_dict_test = joblib.load('data/{}/indiv_test_prob.pkl'.format(fea))
        dialogs = dialogs_test.copy()
        emo_dict = emo_dict_test.copy()
        out_dict = out_dict_test.copy()
    
    # dialogs.update(dialogs_test)
    # emo_dict.update(emo_dict_test)
    # out_dict.update(out_dict_test)
    
    recording_lst = sorted(set([i.split('_')[0] for i in dialogs.keys()]))
    performance = pd.DataFrame(columns=['recording', 'subjectPos1', 'subjectPos2', 'subjectPos3', 'subjectPos4'])
    performance['recording']=recording_lst
    performance.index = performance['recording']
    # pdb.set_trace()
    '''
    dialog:
        key: recording
        value: who talk
    
    
    out_dict, emo_dict:
        key-> 10 sec index sequence
        value -> talk_tend (dominet -> least)
    '''
    
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
    indiv_output = {}
    DED = bs.BeamSearch(p_0, args.crp_alpha, args.num_state, 
                                args.beam_size, args.test_iteration, emo_dict, out_dict, 'no')
    
    for i, dia in enumerate(dialogs):
        logging.info("Decoding dialog: {}/{}, {}".format(i,len(dialogs),dia))
      
        # pdb.set_trace()
        # Apply p_0 estimated from other 4 sessions.
        if args.transition_bias < 0:
            DED.transition_bias = bias_dict[dia[:13]] 
        
        # Beam search decoder
        out, memo = DED.decode(dialogs[dia]) 
                
        # pdb.set_trace()
        trace += out[0]
        label += [utils.convert_to_index_individual_label(emo_dict[utt]) for utt in dialogs[dia]]
        org_pred += [np.argmax(out_dict[utt]) for utt in dialogs[dia]]
        if args.verbosity > 0:
            logging.info("Output: {}\n".format(out))
        
        # temp_label = [utils.convert_to_index_individual_label(emo_dict[utt]) for utt in dialogs[dia]]
        # uar = recall_score(temp_label, out[0], average='macro')
        # precision = precision_score(temp_label, out[0], average = 'macro')
        # performance.loc[dia.split('_')[0], 'subjectPos{}'.format(dia.split('_')[-1])] = uar
        # pdb.set_trace()
        
        indiv_output[dia+'_pred'] = out[0] 
        indiv_output[dia+'_gt'] = [utils.convert_to_index_individual_label(emo_dict[utt]) for utt in dialogs[dia]]         
        indiv_output[dia+'_org'] = [np.argmax(out_dict[utt]) for utt in dialogs[dia]]   
    
    # performance.to_csv('individual_performance.csv')
    
    print("#"*50+"\n")
    print(mode)
    # Print the results of emotino classifier module
    acc, uar, precision, f1, conf = utils.evaluate(org_pred, label)
    logging.info('Original performance: uar: %.4f, acc: %.4f, f1 score: %.4f, precision: %.4f' % (uar, acc, f1, precision))
    
    # Eval ded outputs
    
    # pdb.set_trace()
    acc, uar, precision, f1, conf = utils.evaluate(trace, label)
    logging.info('DED individual performance: uar: %.4f, acc: %.4f, f1 score: %.4f, precision: %.4f' % (uar, acc, f1, precision))
    logging.info('Confusion matrix:\n%s' % conf)
    
    
    # Save results
    # results = vars(args)
    # results['uar'] = uar
    # results['acc'] = acc
    # results['conf'] = str(conf)
    # logging.info('Save results:')
    # logging.info('\n%s\n', json.dumps(results, sort_keys=False, indent=4))
    # json.dump(results, open(args.out_dir+'/%s.json' % args.result_file, "w"))
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
                        default=0.153, 
                        help='Transition bias, if not given, p_0 will be estimated from'
                             'training data. See eq (5)')
    parser.add_argument('--num_state', 
                        type=int, 
                        default=2, 
                        help='Number of emotion states. Note that the states are bounded.')
    parser.add_argument('--test_iteration', 
                        type=int, 
                        default=2, 
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
    
    # main('test', 'talk_other/cv4', args)
    outputs = main('test', 'talk_other/cv4', args)
    joblib.dump(outputs, 'rescore_dsd_dic.pkl')
    member_set = sorted(set(['_'.join(i.split('_')[:2]) for i in outputs.keys()]))
    
    transit_pred_lst = []
    transit_gt_lst = []
    transit_org_lst = []
    
    same_org_lst = []
    same_pred_lst = []
    same_gt_lst = []
    
    trans_dic = {}
    same_dic = {}
    for member in member_set:
        ground_truth = outputs[member+'_gt']
        pred = outputs[member+'_pred']
        org = outputs[member+'_org']
        # pdb.set_trace()
        transit_ind = [i for i in range(1, len(ground_truth)) if ground_truth[i]!=ground_truth[i-1] ]
        same_ind = [i for i in range(0, len(ground_truth)) if i not in transit_ind]
        
        trans_dic[member] = transit_ind
        same_dic[member] = same_ind
        
        transit_gt = [ground_truth[i] for i in transit_ind]
        transit_pred = [pred[i] for i in transit_ind]
        transit_org = [org[i] for i in transit_ind]
        
        transit_gt_lst.extend(transit_gt)
        transit_pred_lst.extend(transit_pred)
        transit_org_lst.extend(transit_org)
        
        same_gt = [ground_truth[i] for i in same_ind]
        same_pred = [pred[i] for i in same_ind]
        same_org = [org[i] for i in same_ind]
        
        same_gt_lst.extend(same_gt)
        same_pred_lst.extend(same_pred)
        same_org_lst.extend(same_org)
        
    print('transition count: ', len(transit_gt_lst))
    acc, uar, precision, f1, conf = utils.evaluate(transit_pred_lst, transit_gt_lst)
    print('Transit performance: uar: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}'.format(uar, acc, f1, precision))
    print(classification_report(transit_pred_lst, transit_gt_lst))
    
    acc, uar, precision, f1, conf = utils.evaluate(transit_org_lst, transit_gt_lst)
    print('Transit origin performance: uar: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}'.format(uar, acc, f1, precision))
    print(classification_report(transit_org_lst, transit_gt_lst))
 
    print('same count: ', len(same_gt_lst))
    acc, uar, precision, f1, conf = utils.evaluate(same_pred_lst, same_gt_lst)
    print('Same performance: uar: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}'.format(uar, acc, f1, precision))
    print(classification_report(same_pred_lst, same_gt_lst))

    acc, uar, precision, f1, conf = utils.evaluate(same_org_lst, same_gt_lst)
    print('Same origin performance: uar: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}'.format(uar, acc, f1, precision))
    print(classification_report(same_org_lst, same_gt_lst))
  
  
