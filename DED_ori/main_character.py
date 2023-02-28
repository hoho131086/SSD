#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:29:37 2022

@author: shaohao
"""

import beam_search as bs
from arguement import parse_args
import utils
import os
import sys
import numpy as np
import joblib
import logging
import json
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
import pdb
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(memo_thresh, candi_thresh, mode):
    """
    dialogs: dict, dialogs of the dataset.
    emo_dict: dict, emotions of all utterances.
    out_dict: dict, output logits of emotion classifier.
    """
    if mode == 'train':
        dialogs_train = joblib.load('data/character_train_dialog.pkl')
        emo_dict_train = joblib.load('data/character_train_talk.pkl')
        out_dict_train = joblib.load('data/character_train_prob.pkl')
        
        dialogs = dialogs_train.copy()
        emo_dict = emo_dict_train.copy()
        out_dict = out_dict_train.copy()
    elif mode == 'test':
        dialogs_test = joblib.load('data/character_test_dialog.pkl')
        emo_dict_test = joblib.load('data/character_test_talk.pkl')
        out_dict_test = joblib.load('data/character_test_prob.pkl')
        
        dialogs = dialogs_test.copy()
        emo_dict = emo_dict_test.copy()
        out_dict = out_dict_test.copy()
    
    # dialogs = dialogs_train.copy()
    # dialogs.update(dialogs_test)
    
    # emo_dict = emo_dict_train.copy()
    # emo_dict.update(emo_dict_test)
    
    # out_dict = out_dict_train.copy()
    # out_dict.update(out_dict_test)
    
    recording_lst = sorted(set([i.split('_')[0] for i in dialogs.keys()]))
    performance = pd.DataFrame(columns=['recording', 'talk_most', 'talk_second', 'talk_third', 'talk_least'])
    performance['recording']=recording_lst

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
    args = parse_args()
    
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
    # candi_trace = []
    thresh_trace = []
    label = []
    org_pred = []
    DED = bs.BeamSearch(p_0, args.crp_alpha, args.num_state, 
                                args.beam_size, args.test_iteration, emo_dict, out_dict, 'yes')
    
    for i, dia in enumerate(dialogs):
        logging.info("Decoding dialog: {}/{}, {}".format(i,len(dialogs),dia))
      
        # pdb.set_trace()
        # Apply p_0 estimated from other 4 sessions.
        if args.transition_bias < 0:
            DED.transition_bias = bias_dict[dia[:11]] 
        
        # Beam search decoder
        out, memo = DED.decode(dialogs[dia]) 
    
        # memo_thresh = 2
        out_thresh = utils.convert_to_index_memo(out[0], memo, memo_thresh) # 0.722
        
        # candi_thresh = 3
        # out_candi = utils.convert_to_index_candidate(out, candi_thresh)
        
        
        pdb.set_trace()
        thresh_trace += out_thresh
        # candi_trace += out_candi
        label += [utils.convert_to_index(emo_dict[utt]) for utt in dialogs[dia]]
        org_pred += utils.convert_to_index_out([np.argmax(out_dict[utt]) for utt in dialogs[dia]])
        if args.verbosity > 0:
            logging.info("Output: {}\n".format(out))
            
        a = np.array(out_thresh)
        b = np.array([utils.convert_to_index(emo_dict[utt]) for utt in dialogs[dia]])
        for ind in range(4):
            recall = recall_score(b[:,ind], a[:,ind], average='macro')
            precision = precision_score(b[:,ind], a[:,ind], average = 'macro')
            uar = (recall+precision)/2.0
            
            performance.iloc[i, ind+1] = uar

        # pdb.set_trace()
    
    performance.to_csv('origin_performance_{}.csv'.format(memo_thresh))
    
    print("#"*50+"\n")
    # Print the results of emotino classifier module
    acc, uar, precision, f1, conf = utils.evaluate(org_pred, label)
    logging.info('Original performance: uar: %.3f, acc: %.3f, f1 score: %.3f, precision: %.3f' % (uar, acc, f1, precision))
    
    # Eval ded outputs
    
    # pdb.set_trace()
    acc, uar, precision, f1, conf = utils.evaluate(thresh_trace, label)
    logging.info('DED memo thresh: %.2f performance: uar: %.3f, acc: %.3f, f1 score: %.3f, precision: %.3f' % (memo_thresh, uar, acc, f1, precision))
    logging.info('Confusion matrix:\n%s' % conf)
    
    # acc, uar, precision, recall, conf = utils.evaluate(candi_trace, label)
    # logging.info('DED candidate %.2f performance: uar: %.3f, acc: %.3f, recall: %.3f, precision: %.3f' % (candi_thresh, uar, acc, recall, precision))
    # logging.info('Confusion matrix:\n%s' % conf)
    
    # Save results
    # results = vars(args)
    # results['uar'] = uar
    # results['acc'] = acc
    # results['conf'] = str(conf)
    # logging.info('Save results:')
    # logging.info('\n%s\n', json.dumps(results, sort_keys=False, indent=4))
    # json.dump(results, open(args.out_dir+'/%s.json' % args.result_file, "w"))


if __name__ == '__main__':  
    # main(memo_thresh = 0, candi_thresh=0)
    main(memo_thresh = 0, candi_thresh=0, mode = 'test')
    main(memo_thresh = 0.5, candi_thresh=0, mode = 'test')
    main(memo_thresh = 1, candi_thresh=0, mode = 'test')
    # main(memo_thresh = -1, candi_thresh=2)
    # main(memo_thresh = 0, candi_thresh=1)

    
  
  
  
  
