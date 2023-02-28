#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:38:50 2022

@author: shaohao
"""


import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import MultiheadAttention
import pdb
import math


class GATE_talk(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(GATE_talk, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input = nn.Sequential(*network)
                
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input = nn.Sequential(*network)       
        
        r_talk_candi_size = (self.talk_hidden+self.me_gaze_hidden)
        r_talk_candi = [nn.Linear(r_talk_candi_size, r_talk_candi_size), nn.ReLU(),
                        nn.Linear(r_talk_candi_size, r_talk_candi_size), nn.Sigmoid()]
        
        self.r_talk_candi_layer = nn.Sequential(*r_talk_candi)
        
        r_talk_other_size = (self.talk_hidden+self.gaze_hidden)
        r_talk_other = [nn.Linear(r_talk_other_size, r_talk_other_size), nn.ReLU(),
                        nn.Linear(r_talk_other_size, r_talk_other_size), nn.Sigmoid()]
        
        self.r_talk_other_layer = nn.Sequential(*r_talk_other)
        
        gate_size = (self.talk_hidden+self.gaze_hidden)*2
        gate_net = [nn.Linear(gate_size, int(gate_size/2)), nn.Sigmoid()]
        self.gate_layer = nn.Sequential(*gate_net)
        
        final_embed = r_talk_other_size
        network = [nn.Linear(final_embed, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.talk_input(talk)
        other_gaze = torch.cat([right, middle, left], dim = 2)
        other_gaze = self.gaze_input(other_gaze)
        me_gaze = self.me_gaze_input(candi)
        # pdb.set_trace()
        r_talk_candi = torch.cat([talk, me_gaze], dim = 2)
        r_talk_candi = torch.mean(r_talk_candi, dim=1)
        r_talk_candi = self.r_talk_candi_layer(r_talk_candi)
        
        r_talk_other = torch.cat([talk, other_gaze], dim = 2)
        r_talk_other = torch.mean(r_talk_other, dim=1)
        r_talk_other = self.r_talk_other_layer(r_talk_other)
        
        gate_input = torch.cat([r_talk_candi, r_talk_other],dim=1)
        gate = self.gate_layer(gate_input)
        
        
        final_input = r_talk_candi+(1-gate)*r_talk_other
        
        final_input = self.clf(final_input)
        
        return final_input

class GATE_other(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(GATE_other, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input = nn.Sequential(*network)
                
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input = nn.Sequential(*network)       
        
        r_other_candi_size = (self.gaze_hidden+self.me_gaze_hidden)
        r_other_candi = [nn.Linear(r_other_candi_size, r_other_candi_size), nn.ReLU(),
                        nn.Linear(r_other_candi_size, r_other_candi_size), nn.Sigmoid()]
        
        self.r_other_candi_layer = nn.Sequential(*r_other_candi)
        
        r_other_talk_size = (self.talk_hidden+self.gaze_hidden)
        r_other_talk = [nn.Linear(r_other_talk_size, r_other_talk_size), nn.ReLU(),
                        nn.Linear(r_other_talk_size, r_other_talk_size), nn.Sigmoid()]
        
        self.r_other_talk_layer = nn.Sequential(*r_other_talk)
        
        gate_size = (self.talk_hidden+self.gaze_hidden)*2
        gate_net = [nn.Linear(gate_size, int(gate_size/2)), nn.Sigmoid()]
        self.gate_layer = nn.Sequential(*gate_net)
        
        final_embed = r_other_talk_size
        network = [nn.Linear(final_embed, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.talk_input(talk)
        other_gaze = torch.cat([right, middle, left], dim = 2)
        other_gaze = self.gaze_input(other_gaze)
        me_gaze = self.me_gaze_input(candi)
        # pdb.set_trace()
        r_other_candi = torch.cat([other_gaze, me_gaze], dim = 2)
        r_other_candi = torch.mean(r_other_candi, dim=1)
        r_other_candi = self.r_other_candi_layer(r_other_candi)
        
        r_other_talk = torch.cat([other_gaze, talk], dim = 2)
        r_other_talk = torch.mean(r_other_talk, dim=1)
        r_other_talk = self.r_other_talk_layer(r_other_talk)
        
        gate_input = torch.cat([r_other_candi, r_other_talk],dim=1)
        gate = self.gate_layer(gate_input)
        
        
        final_input = r_other_candi+(1-gate)*r_other_talk
        
        final_input = self.clf(final_input)
        
        return final_input


class GATE_candi(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(GATE_candi, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input = nn.Sequential(*network)
                
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input = nn.Sequential(*network)       
        
        r_candi_talk_size = (self.talk_hidden+self.me_gaze_hidden)
        r_candi_talk = [nn.Linear(r_candi_talk_size, r_candi_talk_size), nn.ReLU(),
                        nn.Linear(r_candi_talk_size, r_candi_talk_size), nn.Sigmoid()]
        
        self.r_candi_talk_layer = nn.Sequential(*r_candi_talk)
        
        r_candi_other_size = (self.me_gaze_hidden+self.gaze_hidden)
        r_candi_other = [nn.Linear(r_candi_other_size, r_candi_other_size), nn.ReLU(),
                         nn.Linear(r_candi_other_size, r_candi_other_size), nn.Sigmoid()]
        
        self.r_candi_other_layer = nn.Sequential(*r_candi_other)
        
        gate_size = (self.talk_hidden+self.gaze_hidden)*2
        gate_net = [nn.Linear(gate_size, int(gate_size/2)), nn.Sigmoid()]
        self.gate_layer = nn.Sequential(*gate_net)
        
        final_embed = r_candi_other_size
        network = [nn.Linear(final_embed, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.talk_input(talk)
        other_gaze = torch.cat([right, middle, left], dim = 2)
        other_gaze = self.gaze_input(other_gaze)
        me_gaze = self.me_gaze_input(candi)
        # pdb.set_trace()
        r_candi_talk = torch.cat([me_gaze, talk], dim = 2)
        r_candi_talk = torch.mean(r_candi_talk, dim=1)
        r_candi_talk = self.r_candi_talk_layer(r_candi_talk)
        
        r_candi_other = torch.cat([me_gaze, other_gaze], dim = 2)
        r_candi_other = torch.mean(r_candi_other, dim=1)
        r_candi_other = self.r_candi_other_layer(r_candi_other)
        
        gate_input = torch.cat([r_candi_talk, r_candi_other],dim=1)
        gate = self.gate_layer(gate_input)
        
        
        final_input = r_candi_talk+(1-gate)*r_candi_other
        
        final_input = self.clf(final_input)
        
        return final_input

#%%
if __name__=='__main__':
    
    # dnn feature
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # batch_size = 16
    # feature_size_list = [7,512,768]
    # enc_list = [512,128,128,3]
    # fea_list = [torch.rand(batch_size, feature_size).to(device) for feature_size in feature_size_list]
    # # dnn test
    # dnn_fusion = multi_input_fusion_DNN(feature_size_list, enc_list, device)
    # op = dnn_fusion.forward(fea_list)
    
    # # lstm feature
    # batch_size = 16
    # feature_size_list = [7,512,768]
    # enc_list = [512,128,128,3]
    # time_step = 5
    # fea_list = [torch.rand(batch_size, time_step, feature_size).to(device) for feature_size in feature_size_list]
    # seq_lengths = torch.LongTensor([time_step for i in range(batch_size)])
    # # lstm test
    # lstm_fusion = multi_input_fusion_BiLSTM(feature_size_list, enc_list, device)
    # op = lstm_fusion.forward(fea_list, seq_lengths)

    # tfm feature
    # batch_size = 16
    # feature_size = 7
    # enc_list = [512,128,128,1]
    # time_step = 5
    # fea = torch.rand(batch_size, time_step, feature_size).to(device)
    # seq_lengths = torch.LongTensor([time_step for i in range(batch_size)])
    # # tfm test(self, feat_dim, hidden_dim, hidden_layers_num, cl_num, tfm_head, dropout_r, device)
    # tfm_fusion = Trans_Encoder_clf(feature_size, 64, 2, 3, 8, 0.3, device).to(device)
    # op = tfm_fusion.forward(fea, seq_lengths)
    
    # batch_size = 16
    # feature_size = 16
    # time_step = 5
    # fea = torch.rand(batch_size, time_step, feature_size).to(device)
    # seq_lengths = torch.LongTensor([time_step for i in range(batch_size)])
    # # tfm test(self, feat_dim, hidden_dim, hidden_layers_num, cl_num, tfm_head, dropout_r, device)
    # att = self_Attn(hidden_size=16, head=8).to(device)
    # op = att.forward(fea)
    
    
    # batch_size = 16
    # feature_size = 12
    # hidden = 24
    # time_step = 30
    # talk = torch.rand(batch_size, time_step, 1).to(device)
    # fea = torch.rand(batch_size, time_step, feature_size).to(device)
    # other = torch.rand(batch_size, time_step, feature_size*3).to(device)
    # # (self, fea_dim, tfm_head, out_dim)
    # att = ATT(feature_size, hidden, 8, 1, dropout=0.5).to(device)
    
    # op = att.forward(talk, fea, fea, fea, fea)
    
    
    batch_size = 16
    feature_size = 12
    hidden = 24
    time_step = 30
    talk = torch.rand(batch_size, time_step, 1).to(device)
    fea = torch.rand(batch_size, time_step, feature_size).to(device)
    other = torch.rand(batch_size, time_step, feature_size*3).to(device)
    # (self, fea_dim, tfm_head, out_dim)
    # att = ATT_John(feature_size, hidden, 8, 1, 0.5, 'nn').to(device)
    
    # op = att.forward(talk, fea, fea, fea, fea)
    
    
    