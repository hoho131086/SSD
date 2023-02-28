#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 23:00:15 2022

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

class ATT_TALK(nn.Module): ## GOOD performance need weight sampler
    def __init__(self, fea_dim, hidden_dim, layer_num, tfm_head, out_dim):
        super(ATT_TALK, self).__init__()
        
        network = [nn.Linear(fea_dim, hidden_dim), nn.ReLU()]
        self.input_layer = nn.Sequential(*network)
        
        self.layers = layer_num
        self.hidden_size = hidden_dim
        self.head = tfm_head
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size, nhead=self.head, activation='gelu', dropout = 0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers, norm=nn.LayerNorm(self.hidden_size))
        # self.final = nn.Sigmoid()
        network = [nn.Linear(hidden_dim, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.input_layer(talk)
        talk = talk.transpose(0, 1)
        talk = self.transformer_encoder.forward(talk)
        talk = talk.transpose(0, 1)
        
        talk = self.clf(torch.mean(talk, dim = 1))
        # pdb.set_trace()
        return talk

class ATT_OTHER_GAZE(nn.Module): 
    def __init__(self, fea_dim, hidden_dim, layer_num, tfm_head, out_dim, dropout, func):
        super(ATT_OTHER_GAZE, self).__init__()
        
        network = [nn.Linear(fea_dim*3, hidden_dim), nn.ReLU()]
        self.input_layer = nn.Sequential(*network)
        
        self.layers = layer_num
        self.hidden_size = hidden_dim
        self.head = tfm_head
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size, nhead=self.head, activation='gelu', dropout = 0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers, norm=nn.LayerNorm(self.hidden_size))
        # self.final = nn.Sigmoid()
        network = [nn.Linear(hidden_dim, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        other_gaze = torch.cat([right, middle, left], dim = 2)
        # pdb.set_trace()
        other_gaze = self.input_layer(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        other_gaze = self.transformer_encoder.forward(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        
        other_gaze = self.clf(torch.mean(other_gaze, dim = 1))
        return other_gaze
    
class ATT_ME_GAZE(nn.Module): 
    def __init__(self, fea_dim, hidden_dim, layer_num, tfm_head, out_dim):
        super(ATT_ME_GAZE, self).__init__()
        
        network = [nn.Linear(fea_dim, hidden_dim), nn.ReLU()]
        self.input_layer = nn.Sequential(*network)
        
        self.layers = layer_num
        self.hidden_size = hidden_dim
        self.head = tfm_head
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size, nhead=self.head, activation='gelu', dropout = 0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers, norm=nn.LayerNorm(self.hidden_size))
        # self.final = nn.Sigmoid()
        network = [nn.Linear(hidden_dim, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        candi = self.input_layer(candi)
        candi = candi.transpose(0, 1)
        candi = self.transformer_encoder.forward(candi)
        candi = candi.transpose(0, 1)
        
        candi = self.clf(torch.mean(candi, dim = 1))
        # pdb.set_trace()
        return candi

class ATT_Combine(nn.Module):  # talk other
    def __init__(self, talk_fea, talk_hidden, gaze_fea, gaze_hidden, layer_num, tfm_head_talk, tfm_head_gaze, out_dim):
        super(ATT_Combine, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input = nn.Sequential(*network)
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input = nn.Sequential(*network)
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))

        network = [nn.Linear(gaze_hidden+talk_hidden, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.talk_input(talk)
        talk = talk.transpose(0, 1)
        talk = self.transformer_encoder_talk.forward(talk)
        talk = talk.transpose(0, 1)
        
        other_gaze = torch.cat([right, middle, left], dim = 2)
        # pdb.set_trace()
        other_gaze = self.gaze_input(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        other_gaze = self.transformer_encoder_gaze.forward(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        
        final_input = torch.cat([talk, other_gaze], dim = 2)
        
        final_input = self.clf(torch.mean(final_input, dim = 1))
        
        return final_input

class ATT_Combine_Me(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_Combine_Me, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input = nn.Sequential(*network)
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input = nn.Sequential(*network)
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input = nn.Sequential(*network)
        
        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))

        
        network = [nn.Linear(gaze_hidden+talk_hidden+me_gaze_hidden, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.talk_input(talk)
        talk = talk.transpose(0, 1)
        talk = self.transformer_encoder_talk.forward(talk)
        talk = talk.transpose(0, 1)
        
        other_gaze = torch.cat([right, middle, left], dim = 2)
        # pdb.set_trace()
        other_gaze = self.gaze_input(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        other_gaze = self.transformer_encoder_gaze.forward(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        
        me_gaze = self.me_gaze_input(candi)
        me_gaze = me_gaze.transpose(0, 1)
        me_gaze = self.transformer_encoder_me_gaze.forward(me_gaze)
        me_gaze = me_gaze.transpose(0, 1)
        
        final_input = torch.cat([talk, other_gaze, me_gaze], dim = 2)
        
        final_input = self.clf(torch.mean(final_input, dim = 1))
        
        return final_input

class ATT_ME_OTHER(nn.Module):
    def __init__(self, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, layer_num, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_ME_OTHER, self).__init__()
        
        self.layers = layer_num
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
                
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input = nn.Sequential(*network)
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input = nn.Sequential(*network)
        
        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))

        
        network = [nn.Linear(gaze_hidden+me_gaze_hidden, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        
        other_gaze = torch.cat([right, middle, left], dim = 2)
        # pdb.set_trace()
        other_gaze = self.gaze_input(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        other_gaze = self.transformer_encoder_gaze.forward(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        
        me_gaze = self.me_gaze_input(candi)
        me_gaze = me_gaze.transpose(0, 1)
        me_gaze = self.transformer_encoder_me_gaze.forward(me_gaze)
        me_gaze = me_gaze.transpose(0, 1)
        
        final_input = torch.cat([other_gaze, me_gaze], dim = 2)
        
        final_input = self.clf(torch.mean(final_input, dim = 1))
        
        return final_input

class ATT_Talk_Me(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, layer_num, tfm_head_talk, head_me_gaze, out_dim):
        super(ATT_Talk_Me, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.me_gaze_hidden = me_gaze_hidden
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input = nn.Sequential(*network)
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input = nn.Sequential(*network)
        
        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))

        
        network = [nn.Linear(talk_hidden+me_gaze_hidden, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.talk_input(talk)
        talk = talk.transpose(0, 1)
        talk = self.transformer_encoder_talk.forward(talk)
        talk = talk.transpose(0, 1)
        
        me_gaze = self.me_gaze_input(candi)
        me_gaze = me_gaze.transpose(0, 1)
        me_gaze = self.transformer_encoder_me_gaze.forward(me_gaze)
        me_gaze = me_gaze.transpose(0, 1)
        
        final_input = torch.cat([talk, me_gaze], dim = 2)
        
        final_input = self.clf(torch.mean(final_input, dim = 1))
        
        return final_input


class ATT_Combine_Me_analysis(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_Combine_Me_analysis, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input = nn.Sequential(*network)
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input = nn.Sequential(*network)
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input = nn.Sequential(*network)
        
        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))

        
        network = [nn.Linear(gaze_hidden+talk_hidden+me_gaze_hidden, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.talk_input(talk)
        talk = talk.transpose(0, 1)
        talk = self.transformer_encoder_talk.forward(talk)
        talk = talk.transpose(0, 1)
        
        other_gaze = torch.cat([right, middle, left], dim = 2)
        # pdb.set_trace()
        other_gaze = self.gaze_input(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        other_gaze = self.transformer_encoder_gaze.forward(other_gaze)
        other_gaze = other_gaze.transpose(0, 1)
        
        me_gaze = self.me_gaze_input(candi)
        me_gaze = me_gaze.transpose(0, 1)
        me_gaze = self.transformer_encoder_me_gaze.forward(me_gaze)
        me_gaze = me_gaze.transpose(0, 1)
        
        final_input = torch.cat([talk, other_gaze, me_gaze], dim = 2)
        
        final_input = self.clf(torch.mean(final_input, dim = 1))
        
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
    
    
    