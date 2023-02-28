#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:38:12 2022

@author: shaohao
"""

import numpy as np
import utils
import pdb
from itertools import chain

class BeamState(object):

    def __init__(self, spk_sequence, emo_sequence, log_prob, block_counts, memo_sequence):
        """hypothesis
        Args:
            spk_sequence: List, sequence of speaker indexes in dialogical order.
            emo_sequence: List, sequence of emotion state indexes in dialogical order.
            block_counts: List, initially 0 block for each emotion state, note that states are bounded.
        """
        self.spk_sequence = spk_sequence
        self.emo_sequence = emo_sequence
        self.log_prob = log_prob
        self.block_counts = block_counts
        self.memo_sequence = memo_sequence

    def update(self, spk, state, log_prob, memo):
        """Return new beam state based on last decoding results."""
        return  BeamState(
                          self.spk_sequence+[spk],
                          self.emo_sequence+[state],
                          self.log_prob+log_prob,
                          self.block_counts,
                          self.memo_sequence+[memo])
   
    def copy_beam(self):
        """Return copied beam state."""
        return  BeamState(
                          self.spk_sequence.copy(),
                          self.emo_sequence.copy(),
                          self.log_prob,
                          self.block_counts.copy(),
                          self.memo_sequence.copy())

class BeamSearch(object):

    def __init__(self, bias, crp_alpha, num_state, beam_size, test_iteration, emo, logits, use_memo):
    # Define parameters
        self.transition_bias = bias
        self.crp_alpha = crp_alpha
        self.num_state = num_state
        self.beam_size = beam_size
        self.test_iteration = test_iteration
        self.emo = emo
        self.all_logits = logits
        self.use_memo = use_memo

    def decode(self, dialog):
        """
        Args:
            dialog: List, a list of utterances id in dialogical order.
      
            For example:
            ```
            dialog = 
            ['Ses01M_XX_M000', 'Ses01M_XX_F000', 'Ses01M_XX_F002', ...]
            ```
        Returns:
            predicted_sequence: Predicted emotion state sequence. An array of integers.
        """
        
        # pdb.set_trace()
        test_sequence_length = len(dialog)
        test_sequence = np.tile(dialog, self.test_iteration)
                
        # decoding steps
        dec_step = len(test_sequence)
        t = 0
        beam_set = [BeamState([], [], 0, [0 for i in range(self.num_state)], []) for _ in range(self.beam_size)]
        selected_beam_state = []
        while t < dec_step and len(selected_beam_state) < self.beam_size:
            utt_id = test_sequence[t]
            beam_set_bank = []
            
            num_beam = len(beam_set) if t > 0 else 1
            # pdb.set_trace()
            # collect nodes
            for i in range(num_beam):
                for j in range(self.num_state):  # => search each node
                    updated_beam = self._update_beam(beam_set[i].copy_beam(), utt_id, j)
                    beam_set_bank.append(updated_beam)
            
            # sort by log prob
            beam_set = self._select_best_k(beam_set_bank)
            t += 1
        # pdb.set_trace()
        
        return (beam_set[0].emo_sequence[-len(dialog):], beam_set[1].emo_sequence[-len(dialog):],
                beam_set[2].emo_sequence[-len(dialog):], beam_set[3].emo_sequence[-len(dialog):], beam_set[4].emo_sequence[-len(dialog):]), beam_set[0].memo_sequence[-len(dialog):]

    def _get_logits(self, utt_id):
        return self.all_logits[utt_id]

    def _update_beam(self, beam_state, utt_id, state):
        """Calculate log probability for shift and assignment process based on current beam state."""
      
        loss = 0
        speaker = utt_id[-4]
        if speaker!='_':
            print(speaker, utt_id)
        # pdb.set_trace()
        # Convert the state into one-hot vector
        label = np.zeros(self.num_state)
        label[state] = 1
        # Get original ce loss
        logit = np.reshape(self.all_logits[utt_id], [self.num_state,])
        loss = utils.cross_entropy(label, logit)
        # pdb.set_trace()
        ####### find candidate #########
        if self.use_memo == 'yes':
            all_candidate_loss = {}
            for i in range(self.num_state):
                candidate_label = np.zeros(self.num_state)
                candidate_label[i] = 1
                # Get original ce loss
                candidate_loss = utils.cross_entropy(candidate_label, logit)
                # pdb.set_trace()
                if i in np.unique(beam_state.emo_sequence) and speaker in np.unique(beam_state.spk_sequence): 
    
                    last_idx = utils.find_last_idx(beam_state.spk_sequence, speaker)
                    last_state = beam_state.emo_sequence[last_idx]
                    if i == last_state:
                        candidate_loss -= np.log(1 - self.transition_bias)
                    # Shift
                    else:
                        candidate_loss -= np.log(self.transition_bias) + \
                                          np.log(beam_state.block_counts[i]) - \
                                          np.log(sum(beam_state.block_counts) + self.crp_alpha)
                else: 
            
                    candidate_loss -= np.log(self.transition_bias) + \
                                      np.log(self.crp_alpha) - \
                                      np.log(sum(beam_state.block_counts) + self.crp_alpha) 
                # print(i,candidate_loss)
                all_candidate_loss[i]=candidate_loss
          
            ## do thresh
            # all_candidate_loss = {k: v for k, v in sorted(all_candidate_loss.items(), key=lambda item: item[1])}
            memo = all_candidate_loss.copy()
        else:
            memo = []
        
        # RESCORING:
      
        # An existing state
        if state in np.unique(beam_state.emo_sequence) and speaker in np.unique(beam_state.spk_sequence):

          # Find last state
            last_idx = utils.find_last_idx(beam_state.spk_sequence, speaker)
            last_state = beam_state.emo_sequence[last_idx]
            # No shift
            if state == last_state:
                loss -= np.log(1 - self.transition_bias)
            # Shift
            else:
              
                loss -= np.log(self.transition_bias) + \
                        np.log(beam_state.block_counts[state]) - \
                        np.log(sum(beam_state.block_counts) + self.crp_alpha)
                beam_state.block_counts[state] += 1
      
        # A new state
        else: 
      
            loss -= np.log(self.transition_bias) + \
                    np.log(self.crp_alpha) - \
                    np.log(sum(beam_state.block_counts) + self.crp_alpha)          
            beam_state.block_counts[state] += 1
        
        new_beam_state = beam_state.update(speaker, state, loss, memo)
      
        return new_beam_state


    def _select_best_k(self, beam_set):
        """select top k BeamState
        Args:
            beam_set: list, beam set.
        Returns:
            beam_set: list, sorted beam_set based on log_prob in acending order.
        """
        log_prob = [b.log_prob for b in beam_set]
      
        idx = np.argsort(log_prob)[:self.beam_size]
      
        return [beam_set[i] for i in idx]

#%%
class BeamSearch_RealTime(object):

    def __init__(self, bias, crp_alpha, num_state, beam_size, test_iteration, emo, logits, use_memo):
    # Define parameters
        self.transition_bias = bias
        self.crp_alpha = crp_alpha
        self.num_state = num_state
        self.beam_size = beam_size
        self.test_iteration = test_iteration
        self.emo = emo
        self.all_logits = logits
        self.use_memo = use_memo

    def decode(self, t, beam_set, utt_id):
        """
        Args:
            dialog: List, a list of utterances id in dialogical order.
      
            For example:
            ```
            dialog = 
            ['Ses01M_XX_M000', 'Ses01M_XX_F000', 'Ses01M_XX_F002', ...]
            ```
        Returns:
            predicted_sequence: Predicted emotion state sequence. An array of integers.
        """
        
        # test_sequence = np.tile(dialog, self.test_iteration)
        # initial
        if len(beam_set) == 0:
            beam_set = [BeamState([], [], 0, [0 for i in range(self.num_state)], []) for _ in range(self.beam_size)]
        
        # utt_id = test_sequence[t]
        beam_set_bank = []
        
        num_beam = len(beam_set) if t > 0 else 1

        for i in range(num_beam):
            for j in range(self.num_state):  # => search each node
                updated_beam = self._update_beam(beam_set[i].copy_beam(), utt_id, j)
                beam_set_bank.append(updated_beam)
        
        # sort by log prob
        beam_set = self._select_best_k(beam_set_bank)

        select = beam_set[0].emo_sequence[-1]
        # pdb.set_trace()
        
        return beam_set, [select]
    
    def _get_logits(self, utt_id):
        return self.all_logits[utt_id]

    def _update_beam(self, beam_state, utt_id, state):
        """Calculate log probability for shift and assignment process based on current beam state."""
      
        loss = 0
        speaker = utt_id[-4]
        if speaker!='_':
            print(speaker, utt_id)
        # pdb.set_trace()
        # Convert the state into one-hot vector
        label = np.zeros(self.num_state)
        label[state] = 1
        # Get original ce loss
        logit = np.reshape(self.all_logits[utt_id], [self.num_state,])
        loss = utils.cross_entropy(label, logit)
        # pdb.set_trace()
        ####### find candidate #########
        if self.use_memo == 'yes':
            all_candidate_loss = {}
            for i in range(self.num_state):
                candidate_label = np.zeros(self.num_state)
                candidate_label[i] = 1
                # Get original ce loss
                candidate_loss = utils.cross_entropy(candidate_label, logit)
                # pdb.set_trace()
                if i in np.unique(beam_state.emo_sequence) and speaker in np.unique(beam_state.spk_sequence): 
    
                    last_idx = utils.find_last_idx(beam_state.spk_sequence, speaker)
                    last_state = beam_state.emo_sequence[last_idx]
                    if i == last_state:
                        candidate_loss -= np.log(1 - self.transition_bias)
                    # Shift
                    else:
                        candidate_loss -= np.log(self.transition_bias) + \
                                          np.log(beam_state.block_counts[i]) - \
                                          np.log(sum(beam_state.block_counts) + self.crp_alpha)
                else: 
            
                    candidate_loss -= np.log(self.transition_bias) + \
                                      np.log(self.crp_alpha) - \
                                      np.log(sum(beam_state.block_counts) + self.crp_alpha) 
                # print(i,candidate_loss)
                all_candidate_loss[i]=candidate_loss
          
            ## do thresh
            # all_candidate_loss = {k: v for k, v in sorted(all_candidate_loss.items(), key=lambda item: item[1])}
            memo = all_candidate_loss.copy()
        else:
            memo = []
        
        # RESCORING:
      
        # An existing state
        if state in np.unique(beam_state.emo_sequence) and speaker in np.unique(beam_state.spk_sequence):

          # Find last state
            last_idx = utils.find_last_idx(beam_state.spk_sequence, speaker)
            last_state = beam_state.emo_sequence[last_idx]
            # No shift
            if state == last_state:
                loss -= np.log(1 - self.transition_bias)
            # Shift
            else:
              
                loss -= np.log(self.transition_bias) + \
                        np.log(beam_state.block_counts[state]) - \
                        np.log(sum(beam_state.block_counts) + self.crp_alpha)
                beam_state.block_counts[state] += 1
      
        # A new state
        else: 
      
            loss -= np.log(self.transition_bias) + \
                    np.log(self.crp_alpha) - \
                    np.log(sum(beam_state.block_counts) + self.crp_alpha)          
            beam_state.block_counts[state] += 1
        
        new_beam_state = beam_state.update(speaker, state, loss, memo)
      
        return new_beam_state


    def _select_best_k(self, beam_set):
        """select top k BeamState
        Args:
            beam_set: list, beam set.
        Returns:
            beam_set: list, sorted beam_set based on log_prob in acending order.
        """
        log_prob = [b.log_prob for b in beam_set]
      
        idx = np.argsort(log_prob)[:self.beam_size]
      
        return [beam_set[i] for i in idx]
