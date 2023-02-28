#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:16:40 2022

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

class Trans_Encoder_clf(nn.Module):
    def __init__(self, feat_dim, hidden_dim, hidden_layers_num, cl_num, tfm_head, dropout_r, device):
        super(Trans_Encoder_clf, self).__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers_num = hidden_layers_num
        self.cl_num = cl_num
        self.tfm_head = tfm_head
        # self.self_att_head = self_att_head
        # self.max_length = max_length
        
        # input fc
        network = [nn.Linear(self.feat_dim, self.hidden_dim), nn.ReLU()]
        self.input_layer = nn.Sequential(*network)
        
        # tfm_encoder
        self.tfm = TFM(self.hidden_layers_num, self.hidden_dim, self.tfm_head).to(device)
        
        # drop_out
        self.drop_out = nn.Dropout(p=dropout_r)
        
        # output fc
        network = [nn.Linear(self.hidden_dim, self.cl_num), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, input_seqs, hidden=None):
        input_fc_op = self.input_layer.forward(input_seqs)

        # # tmf_emcoder
        input_fc_op = input_fc_op.transpose(0, 1)
        tfm_output = self.tfm.forward(input_fc_op)
        # # output fc
        tfm_output = tfm_output.transpose(0, 1)
        clf_output = self.clf.forward(torch.mean(tfm_output, dim=1))
        # tfm_output = tfm_output.mean(dim=1)
        # pdb.set_trace()  
        
        return clf_output
#%%

class attentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.2):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, src, tar):
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2, weight = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src, weight   

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
        self.att_talk = attentionLayer(self.talk_hidden, self.head_talk)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input = nn.Sequential(*network)
        self.att_gaze = attentionLayer(self.gaze_hidden, self.head_talk)
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input = nn.Sequential(*network)
        self.att_me_gaze = attentionLayer(self.me_gaze_hidden, self.head_me_gaze)
        
        network = [nn.Linear(gaze_hidden+talk_hidden+me_gaze_hidden, out_dim), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.talk_input(talk)
        talk, self.talk_weight = self.att_talk.forward(talk, talk)
        
        other_gaze = torch.cat([right, middle, left], dim = 2)
        # pdb.set_trace()
        other_gaze = self.gaze_input(other_gaze)
        other_gaze, self.gaze_weight = self.att_gaze.forward(other_gaze,other_gaze)
        
        me_gaze = self.me_gaze_input(candi)
        me_gaze, self.me_gaze_weight = self.att_me_gaze.forward(me_gaze,me_gaze)
        
        final_input = torch.cat([talk, other_gaze, me_gaze], dim = 2)
        
        final_input = self.clf(torch.mean(final_input, dim = 1))
        
        return final_input

class ATT(nn.Module):
    def __init__(self, fea_dim, hidden_dim, tfm_head, out_dim, dropout):
        super(ATT, self).__init__()
        
        network = [nn.Linear(fea_dim, hidden_dim), nn.ReLU()]
        self.right_input_layer = nn.Sequential(*network)
        
        network = [nn.Linear(fea_dim, hidden_dim), nn.ReLU()]
        self.middle_input_layer = nn.Sequential(*network)
        
        network = [nn.Linear(fea_dim, hidden_dim), nn.ReLU()]
        self.left_input_layer = nn.Sequential(*network)
        
        network = [nn.Linear(fea_dim, hidden_dim), nn.ReLU()]
        self.me_input_layer = nn.Sequential(*network)

        self.cross_right = attentionLayer(hidden_size = hidden_dim, head = tfm_head, dropout = dropout)
        self.cross_middle = attentionLayer(hidden_size = hidden_dim, head = tfm_head, dropout = dropout)
        self.cross_left = attentionLayer(hidden_size = hidden_dim, head = tfm_head, dropout = dropout)
        
        self.self_att_gaze = attentionLayer(hidden_size = hidden_dim*3, head = tfm_head, dropout = dropout)

        self.FC = nn.Linear(hidden_dim*3+1, out_dim)
        self.final = nn.Sigmoid()

    def forward(self, talk, candi, right, middle, left):
        right_input = self.right_input_layer.forward(right)
        middle_input = self.middle_input_layer.forward(middle)
        left_input = self.left_input_layer.forward(left)
        me_input = self.me_input_layer.forward(candi)
        
        cross_right = self.cross_right(src = me_input, tar = right_input)
        cross_middle = self.cross_middle(src = me_input, tar = middle_input)
        cross_left = self.cross_left(src = me_input, tar = left_input)
        
        x = torch.cat((cross_right, cross_middle, cross_left), 2)
        x = self.self_att_gaze(src = x, tar = x)  
        
        x = torch.cat((talk, x), 2)
        pred = self.FC(torch.mean(x, dim=1))
        pred = self.final(pred)
                
        return pred
#%%
class simple_att(nn.Module):

    def __init__(self, hidden_size, head, dropout=0.1):
        super(simple_att, self).__init__()
        self.self_attn = MultiheadAttention(hidden_size, head, dropout=dropout)
        
    def forward(self, src, tar):
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        
        src = src.transpose(0, 1)
        
        return src  
    
    
class attentionLayer_john(nn.Module):

    def __init__(self, hidden_size, head, class_num, dropout=0.1):
        super(attentionLayer_john, self).__init__()
        self.self_attn = MultiheadAttention(hidden_size, head, dropout=dropout)
        
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        self.linear3 = nn.Linear(hidden_size, class_num)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.activation2 = nn.Sigmoid()    
        
    def forward(self, src, tar):
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.transpose(0, 1) # T, B, C -> B, T, C
        # pdb.set_trace()
        src = self.activation2(self.linear3(src))
        
        return src    

class ATT_John(nn.Module):
    def __init__(self, fea_dim, hidden_dim, tfm_head, out_dim, dropout, func):
        super(ATT_John, self).__init__()
        
        self.func = func
        
        linear_blocks = [nn.Linear(fea_dim*3, 64, bias=True),
                         nn.Linear(64, hidden_dim, bias=True)]
        relu_blocks = [nn.ReLU() for i in range(len(linear_blocks)-1)]
        dropout_blocks = [nn.Dropout(dropout) for i in range(len(linear_blocks)-1)]
        network = []
        for idx, block in enumerate(linear_blocks):
            network.append(block)
            if idx < len(linear_blocks)-1:
                network.append(relu_blocks[idx])
                network.append(dropout_blocks[idx])
        network.append(nn.Sigmoid())
        self.other_gaze_input_layer = nn.Sequential(*network)
        
        network = [nn.Linear(fea_dim, hidden_dim), nn.ReLU()]
        network.append(nn.Sigmoid())
        self.me_gaze_input_layer = nn.Sequential(*network)

        self.cross_att = attentionLayer_john(hidden_size = hidden_dim, head = tfm_head, class_num=out_dim, dropout = dropout)
        
        self.self_att = attentionLayer_john(hidden_size = 1, head = 1, class_num=out_dim, dropout = dropout)
        self.simple = simple_att(hidden_size= 1, head = 1, dropout=dropout)
        self.final = nn.Sigmoid()
        self.talk_thr = nn.ReLU()

    def forward(self, talk, candi, right, middle, left):
        
        if self.func == 'mean':
            other_gaze_embed = (right+middle+left)/3.0
        elif self.func == 'nn':
            other_gaze = torch.cat([right, middle, left], dim = 2)
            other_gaze_embed = self.other_gaze_input_layer(other_gaze)
        
        me_gaze_embed = self.me_gaze_input_layer(candi)
        
        cross_att = self.cross_att(src = me_gaze_embed, tar = other_gaze_embed)
        talk = self.simple(src = talk, tar = talk)
        weighted_sum = torch.mean((talk*cross_att), dim=1)
        weighted_sum = self.final(weighted_sum)
                
        return weighted_sum

class ATT_TALK(nn.Module): ## GOOD performance
    def __init__(self, fea_dim, hidden_dim, tfm_head, out_dim, dropout, func):
        super(ATT_TALK, self).__init__()
        
        network = [nn.Linear(1, 4), nn.ReLU()]
        self.input_layer = nn.Sequential(*network)
        
        self.layers = 1
        self.hidden_size = 4
        self.head = 2
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size, nhead=self.head, activation='gelu', dropout = 0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers, norm=nn.LayerNorm(self.hidden_size))
        # self.final = nn.Sigmoid()
        network = [nn.Linear(4, 1), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk):
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
        network = [nn.Linear(hidden_dim, 1), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        other_gaze = torch.cat([right, middle, left], dim = 2)
        other_gaze_embed = self.input_layer(other_gaze)
    
        other_gaze_embed = other_gaze_embed.transpose(0,1)
        other_gaze_embed = self.transformer_encoder.forward(other_gaze_embed)
        other_gaze_embed = other_gaze_embed.transpose(0,1)
        other_gaze_embed = self.clf(torch.mean(other_gaze_embed, dim = 1))
        # pdb.set_trace()
        return other_gaze_embed

class ATT_ME_GAZE(nn.Module): 
    def __init__(self, fea_dim, hidden_dim, layer_num, tfm_head, out_dim, dropout, func):
        super(ATT_ME_GAZE, self).__init__()
        
        network = [nn.Linear(fea_dim, hidden_dim), nn.ReLU()]
        self.input_layer = nn.Sequential(*network)
        
        self.layers = layer_num
        self.hidden_size = hidden_dim
        self.head = tfm_head
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size, nhead=self.head, activation='gelu', dropout = 0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers, norm=nn.LayerNorm(self.hidden_size))
        # self.final = nn.Sigmoid()
        network = [nn.Linear(hidden_dim, 1), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        gaze = self.input_layer(candi)
    
        gaze = gaze.transpose(0,1)
        gaze = self.transformer_encoder.forward(gaze)
        gaze = gaze.transpose(0,1)
        gaze = self.clf(torch.mean(gaze, dim = 1))
        # pdb.set_trace()
        return gaze

#%%
class TFM(nn.Module):
    def __init__(self, layers, hidden_size, head):
        super(TFM, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.head = head
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size, nhead=self.head, activation='gelu', dropout = 0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers, norm=nn.LayerNorm(self.hidden_size))
    
    def forward(self, encoder_outputs):
        out = self.transformer_encoder.forward(encoder_outputs)
        
        return out
    
class ATT_John2(nn.Module): ## GOOD performance
    def __init__(self, talk_fea_dim, gaze_fea_dim, talk_hidden_dim, gaze_hidden_dim, gaze_layer, out_dim, dropout, func):
        super(ATT_John2, self).__init__()
        
        self.func = func
        
        network = [nn.Linear(talk_fea_dim, talk_hidden_dim), nn.ReLU()]
        self.talk_input_layer = nn.Sequential(*network)
        self.talk_trans = TFM(layers = 1, hidden_size=talk_hidden_dim, head = 2)
        
        network = [nn.Linear(gaze_fea_dim*3, gaze_hidden_dim), nn.ReLU(), nn.Linear(gaze_hidden_dim, 4), nn.ReLU()]
        self.gaze_input_layer = nn.Sequential(*network)
        self.gaze_trans = TFM(layers = gaze_layer, hidden_size=4, head = 2)
        
        network = [nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk, candi, right, middle, left):
        talk = self.talk_input_layer(talk)
        talk = talk.transpose(0, 1)
        talk = self.talk_trans.forward(talk)
        talk = talk.transpose(0, 1)
        
        if self.func == 'mean':
            other_gaze_embed = (right+middle+left)/3.0
        elif self.func == 'nn':
            other_gaze = torch.cat([right, middle, left], dim = 2)
            other_gaze_embed = self.gaze_input_layer(other_gaze)
        
        other_gaze_embed = other_gaze_embed.transpose(0,1)
        other_gaze_embed = self.gaze_trans.forward(other_gaze_embed)
        other_gaze_embed = other_gaze_embed.transpose(0,1)
        
        final_input = torch.cat([talk, other_gaze_embed], dim = 2)
        final_input = self.clf(torch.mean(final_input, dim = 1))
        # pdb.set_trace()
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
    att = ATT_John(feature_size, hidden, 8, 1, 0.5, 'nn').to(device)
    
    op = att.forward(talk, fea, fea, fea, fea)
    
    
    