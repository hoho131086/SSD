#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:44:26 2022

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

class ATT_CHAR_OTHER(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, char_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_CHAR_OTHER, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        self.char_hidden = me_gaze_hidden + char_hidden
        #########################################################################################################
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_now = nn.Sequential(*network)
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden*2, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden*2))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_now = nn.Sequential(*network)
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden*2, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden*2))
        #########################################################################################################
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_bef = nn.Sequential(*network)

        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_now = nn.Sequential(*network)

        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden*2, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden*2))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_now = nn.Sequential(*network)
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char_bef = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char_now = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))

        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden*2, dim_feedforward=self.char_hidden*2, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden*2))

        #########################################################################################################
        # torch.cat([talk_embed, other_gaze_embed, me_gaze_embed, interact_embed], dim = 2)
        overall_fea_size = (self.talk_hidden+self.gaze_hidden+self.me_gaze_hidden+self.char_hidden)*2
        
        if out_dim == 1:
            network = [nn.Linear(overall_fea_size, int(overall_fea_size/2)), nn.ReLU(),
                       nn.Linear(int(overall_fea_size/2), out_dim), nn.Sigmoid()]
        else:
            network = [nn.Linear(overall_fea_size, int(overall_fea_size/2)), nn.ReLU(),
                       nn.Linear(int(overall_fea_size/2), out_dim)]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk_bef, candi_bef, right_bef, middle_bef, left_bef, char_bef,\
                      talk_now, candi_now, right_now, middle_now, left_now, char_now):
        #########################################################################################################
        # pdb.set_trace()
        talk_bef = self.talk_input_bef(talk_bef)
        talk_now = self.talk_input_now(talk_now)
        talk_embed = torch.cat([talk_bef, talk_now], dim = 2)
        
        talk_embed = talk_embed.transpose(0, 1)
        talk_embed = self.transformer_encoder_talk.forward(talk_embed)
        talk_embed = talk_embed.transpose(0, 1)
        #########################################################################################################
   
        other_gaze_bef = torch.cat([right_bef, middle_bef, left_bef], dim = 2)
        other_gaze_now = torch.cat([right_now, middle_now, left_now], dim = 2)
        
        other_gaze_bef = self.gaze_input_bef(other_gaze_bef)
        other_gaze_now = self.gaze_input_now(other_gaze_now)
        other_gaze_embed = torch.cat([other_gaze_bef, other_gaze_now], dim = 2)
        
        other_gaze_embed = other_gaze_embed.transpose(0, 1)
        other_gaze_embed = self.transformer_encoder_gaze.forward(other_gaze_embed)
        other_gaze_embed = other_gaze_embed.transpose(0, 1)
        #########################################################################################################
        
        me_gaze_bef = self.me_gaze_input_bef(candi_bef)
        me_gaze_now = self.me_gaze_input_now(candi_now)
        me_gaze_embed = torch.cat([me_gaze_bef, me_gaze_now], dim = 2)
        
        me_gaze_embed = me_gaze_embed.transpose(0, 1)
        me_gaze_embed = self.transformer_encoder_me_gaze.forward(me_gaze_embed)
        me_gaze_embed = me_gaze_embed.transpose(0, 1)
        #########################################################################################################
        # pdb.set_trace()
        char_gaze_bef = self.character_input_bef(char_bef)
        char_gaze_now = self.character_input_now(char_now)
        interact_bef = torch.cat([me_gaze_bef, char_gaze_bef], dim = 2)
        interact_now = torch.cat([me_gaze_now, char_gaze_now], dim = 2)
        
        interact_bef = interact_bef.transpose(0, 1)
        interact_bef = self.transformer_encoder_char_bef.forward(interact_bef)
        interact_bef = interact_bef.transpose(0, 1)
        
        interact_now = interact_now.transpose(0, 1)
        interact_now = self.transformer_encoder_char_now.forward(interact_now)
        interact_now = interact_now.transpose(0, 1)
        
        interact_embed = torch.cat([interact_bef, interact_now], dim = 2)
        interact_embed = interact_embed.transpose(0, 1)
        interact_embed = self.transformer_encoder_char.forward(interact_embed)
        interact_embed = interact_embed.transpose(0, 1)
        #########################################################################################################
        
        final_embed = torch.cat([talk_embed, other_gaze_embed, me_gaze_embed, interact_embed], dim = 2)
        final_embed = torch.mean(final_embed, dim = 1)
        
        final_input = self.clf(final_embed)
        
        return final_embed, final_input

#%%

class ATT_CHAR_OTHER_V2(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, char_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_CHAR_OTHER_V2, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        self.char_hidden = char_hidden
        #########################################################################################################
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_now = nn.Sequential(*network)
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden*2, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden*2))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_now = nn.Sequential(*network)
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden*2, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden*2))
        #########################################################################################################
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_bef = nn.Sequential(*network)

        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_now = nn.Sequential(*network)

        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden*2, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden*2))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_now = nn.Sequential(*network)
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden*2, dim_feedforward=self.char_hidden*2, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden*2))
        #########################################################################################################
        # torch.cat([talk_embed, other_gaze_embed, me_gaze_embed, interact_embed], dim = 2)
        overall_fea_size = (self.talk_hidden+self.gaze_hidden+self.me_gaze_hidden+self.char_hidden)*2
        
        if out_dim == 1:
            network = [nn.Linear(overall_fea_size, int(overall_fea_size/2)), nn.ReLU(),
                       nn.Linear(int(overall_fea_size/2), out_dim), nn.Sigmoid()]
        else:
            network = [nn.Linear(overall_fea_size, int(overall_fea_size/2)), nn.ReLU(),
                       nn.Linear(int(overall_fea_size/2), out_dim)]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk_bef, candi_bef, right_bef, middle_bef, left_bef, char_bef,\
                      talk_now, candi_now, right_now, middle_now, left_now, char_now):
        #########################################################################################################
        # pdb.set_trace()
        talk_bef = self.talk_input_bef(talk_bef)
        talk_now = self.talk_input_now(talk_now)
        talk_embed = torch.cat([talk_bef, talk_now], dim = 2)
        
        talk_embed = talk_embed.transpose(0, 1)
        talk_embed = self.transformer_encoder_talk.forward(talk_embed)
        talk_embed = talk_embed.transpose(0, 1)
        #########################################################################################################
   
        other_gaze_bef = torch.cat([right_bef, middle_bef, left_bef], dim = 2)
        other_gaze_now = torch.cat([right_now, middle_now, left_now], dim = 2)
        
        other_gaze_bef = self.gaze_input_bef(other_gaze_bef)
        other_gaze_now = self.gaze_input_now(other_gaze_now)
        other_gaze_embed = torch.cat([other_gaze_bef, other_gaze_now], dim = 2)
        
        other_gaze_embed = other_gaze_embed.transpose(0, 1)
        other_gaze_embed = self.transformer_encoder_gaze.forward(other_gaze_embed)
        other_gaze_embed = other_gaze_embed.transpose(0, 1)
        #########################################################################################################
        
        me_gaze_bef = self.me_gaze_input_bef(candi_bef)
        me_gaze_now = self.me_gaze_input_now(candi_now)
        me_gaze_embed = torch.cat([me_gaze_bef, me_gaze_now], dim = 2)
        
        me_gaze_embed = me_gaze_embed.transpose(0, 1)
        me_gaze_embed = self.transformer_encoder_me_gaze.forward(me_gaze_embed)
        me_gaze_embed = me_gaze_embed.transpose(0, 1)
        #########################################################################################################
        # pdb.set_trace()
        char_gaze_bef = self.character_input_bef(char_bef)
        char_gaze_now = self.character_input_now(char_now)
        char_embed = torch.cat([char_gaze_bef, char_gaze_now], dim = 2)
        
        char_embed = char_embed.transpose(0, 1)
        char_embed = self.transformer_encoder_char.forward(char_embed)
        char_embed = char_embed.transpose(0, 1)
        #########################################################################################################
        
        final_embed = torch.cat([talk_embed, other_gaze_embed, me_gaze_embed, char_embed], dim = 2)
        final_embed = torch.mean(final_embed, dim = 1)
        
        final_input = self.clf(final_embed)
        
        return final_embed, final_input

#%%
class ATT_CHAR_OTHER_V3(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, char_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_CHAR_OTHER_V3, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        self.char_hidden = char_hidden
        #########################################################################################################
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_now = nn.Sequential(*network)
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden*2, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden*2))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_now = nn.Sequential(*network)
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden*2, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden*2))
        #########################################################################################################
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_bef = nn.Sequential(*network)

        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_now = nn.Sequential(*network)

        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden*2, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden*2))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_now = nn.Sequential(*network)
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden*2, dim_feedforward=self.char_hidden*2, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden*2))
        #########################################################################################################
        # torch.cat([talk_embed, other_gaze_embed, me_gaze_embed, interact_embed], dim = 2)
        overall_fea_size = (self.talk_hidden+self.gaze_hidden+self.me_gaze_hidden+self.char_hidden)*2
        
        if out_dim == 1:
            network = [nn.Linear(overall_fea_size, int(overall_fea_size/2)), nn.ReLU(),
                       nn.Linear(int(overall_fea_size/2), out_dim), nn.Sigmoid()]
        else:
            network = [nn.Linear(overall_fea_size, int(overall_fea_size/2)), nn.ReLU(),
                       nn.Linear(int(overall_fea_size/2), out_dim)]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk_bef, candi_bef, right_bef, middle_bef, left_bef, char_bef,\
                      talk_now, candi_now, right_now, middle_now, left_now, char_now):
        #########################################################################################################
        # pdb.set_trace()
        talk_bef = self.talk_input_bef(talk_bef)
        talk_now = self.talk_input_now(talk_now)
        talk_embed = torch.cat([talk_bef, talk_now], dim = 2)
        
        talk_embed = talk_embed.transpose(0, 1)
        talk_embed = self.transformer_encoder_talk.forward(talk_embed)
        talk_embed = talk_embed.transpose(0, 1)
        #########################################################################################################
   
        other_gaze_bef = torch.cat([right_bef, middle_bef, left_bef], dim = 2)
        other_gaze_now = torch.cat([right_now, middle_now, left_now], dim = 2)
        
        other_gaze_bef = self.gaze_input_bef(other_gaze_bef)
        other_gaze_now = self.gaze_input_now(other_gaze_now)
        other_gaze_embed = torch.cat([other_gaze_bef, other_gaze_now], dim = 2)
        
        other_gaze_embed = other_gaze_embed.transpose(0, 1)
        other_gaze_embed = self.transformer_encoder_gaze.forward(other_gaze_embed)
        other_gaze_embed = other_gaze_embed.transpose(0, 1)
        #########################################################################################################
        
        me_gaze_bef = self.me_gaze_input_bef(candi_bef)
        me_gaze_now = self.me_gaze_input_now(candi_now)
        me_gaze_embed = torch.cat([me_gaze_bef, me_gaze_now], dim = 2)
        
        me_gaze_embed = me_gaze_embed.transpose(0, 1)
        me_gaze_embed = self.transformer_encoder_me_gaze.forward(me_gaze_embed)
        me_gaze_embed = me_gaze_embed.transpose(0, 1)
        #########################################################################################################
        # pdb.set_trace()
        char_gaze_bef = self.character_input_bef(char_bef)
        char_gaze_now = self.character_input_now(char_now)
        char_embed = torch.cat([char_gaze_bef, char_gaze_now], dim = 2)
        
        char_embed = char_embed.transpose(0, 1)
        char_embed = self.transformer_encoder_char.forward(char_embed)
        char_embed = char_embed.transpose(0, 1)
        #########################################################################################################
        
        final_embed = torch.cat([talk_embed, other_gaze_embed, me_gaze_embed, char_embed], dim = 2)
        final_embed = torch.mean(final_embed, dim = 1)
        
        final_input = self.clf(final_embed)
        
        return final_embed, final_input
#%%


class ATT_NEW(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, char_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_NEW, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        self.char_hidden = char_hidden
        #########################################################################################################
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_now = nn.Sequential(*network)
        
        self.talk_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.talk_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk_bef = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk_now = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_now = nn.Sequential(*network)
        
        self.gaze_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.gaze_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze_bef = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze_now = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        #########################################################################################################
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_bef = nn.Sequential(*network)

        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_now = nn.Sequential(*network)
        
        self.me_gaze_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.me_gaze_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze_bef = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))
        
        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze_now = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_now = nn.Sequential(*network)
        
        self.char_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.char_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char_bef = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char_now = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))
        #########################################################################################################
        # torch.cat([talk_embed, other_gaze_embed, me_gaze_embed, interact_embed], dim = 2)
        overall_fea_size = (self.talk_hidden+self.gaze_hidden+self.me_gaze_hidden+self.char_hidden)
        self.mean_weight_bef = nn.MultiheadAttention(embed_dim=overall_fea_size, num_heads=2, batch_first=True)
        self.mean_weight_now = nn.MultiheadAttention(embed_dim=overall_fea_size, num_heads=2, batch_first=True)

        if out_dim == 1:
            network = [nn.Linear(overall_fea_size, out_dim), nn.Sigmoid()]
        else:
            network = [nn.Linear(overall_fea_size, out_dim)]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk_bef, candi_bef, right_bef, middle_bef, left_bef, char_bef,\
                      talk_now, candi_now, right_now, middle_now, left_now, char_now):
        #########################################################################################################
        # pdb.set_trace()
        talk_bef = self.talk_input_bef(talk_bef)
        talk_now = self.talk_input_now(talk_now)
        
        talk_embed_bef = talk_bef.transpose(0, 1)
        talk_embed_bef = self.transformer_encoder_talk_bef.forward(talk_embed_bef)
        talk_embed_bef = talk_embed_bef.transpose(0, 1)
        talk_embed_bef = talk_embed_bef*self.talk_alpha_bef

        talk_embed_now = talk_now.transpose(0, 1)
        talk_embed_now = self.transformer_encoder_talk_now.forward(talk_embed_now)
        talk_embed_now = talk_embed_now.transpose(0, 1)
        talk_embed_now = talk_embed_now*self.talk_alpha_now
        #########################################################################################################
   
        other_gaze_bef = torch.cat([right_bef, middle_bef, left_bef], dim = 2)
        other_gaze_now = torch.cat([right_now, middle_now, left_now], dim = 2)
        
        other_gaze_bef = self.gaze_input_bef(other_gaze_bef)
        other_gaze_now = self.gaze_input_now(other_gaze_now)
        
        other_gaze_embed_bef = other_gaze_bef.transpose(0, 1)
        other_gaze_embed_bef = self.transformer_encoder_gaze_bef.forward(other_gaze_embed_bef)
        other_gaze_embed_bef = other_gaze_embed_bef.transpose(0, 1)
        other_gaze_embed_bef = other_gaze_embed_bef*self.gaze_alpha_bef
        
        other_gaze_embed_now = other_gaze_now.transpose(0, 1)
        other_gaze_embed_now = self.transformer_encoder_gaze_now.forward(other_gaze_embed_now)
        other_gaze_embed_now = other_gaze_embed_now.transpose(0, 1)
        other_gaze_embed_now = other_gaze_embed_now*self.gaze_alpha_now
        #########################################################################################################
        
        me_gaze_bef = self.me_gaze_input_bef(candi_bef)
        me_gaze_now = self.me_gaze_input_now(candi_now)
        
        me_gaze_embed_bef = me_gaze_bef.transpose(0, 1)
        me_gaze_embed_bef = self.transformer_encoder_me_gaze_bef.forward(me_gaze_embed_bef)
        me_gaze_embed_bef = me_gaze_embed_bef.transpose(0, 1)
        me_gaze_embed_bef = me_gaze_embed_bef*self.me_gaze_alpha_bef
        
        me_gaze_embed_now = me_gaze_now.transpose(0, 1)
        me_gaze_embed_now = self.transformer_encoder_me_gaze_now.forward(me_gaze_embed_now)
        me_gaze_embed_now = me_gaze_embed_now.transpose(0, 1)
        me_gaze_embed_now = me_gaze_embed_now*self.me_gaze_alpha_now
        #########################################################################################################
        # pdb.set_trace()
        char_gaze_bef = self.character_input_bef(char_bef)
        char_gaze_now = self.character_input_now(char_now)
        
        char_embed_bef = char_gaze_bef.transpose(0, 1)
        char_embed_bef = self.transformer_encoder_char_bef.forward(char_embed_bef)
        char_embed_bef = char_embed_bef.transpose(0, 1)
        char_embed_bef = char_embed_bef*self.char_alpha_bef
        
        char_embed_now = char_gaze_now.transpose(0, 1)
        char_embed_now = self.transformer_encoder_char_now.forward(char_embed_now)
        char_embed_now = char_embed_now.transpose(0, 1)
        char_embed_now = char_embed_now*self.char_alpha_now
        #########################################################################################################
        
        final_embed_bef = torch.cat([talk_embed_bef, other_gaze_embed_bef, me_gaze_embed_bef, char_embed_bef], dim = 2)
        final_embed_now = torch.cat([talk_embed_now, other_gaze_embed_now, me_gaze_embed_now, char_embed_now], dim = 2)
        # pdb.set_trace()
        _, time_weights_bef = self.mean_weight_bef(final_embed_bef, final_embed_bef, final_embed_bef)
        time_weights_bef = time_weights_bef.sum(dim=1)
        weighted_bef = torch.mean(final_embed_bef*time_weights_bef.unsqueeze(2), dim=1)
        
        _, time_weights_now = self.mean_weight_now(final_embed_now, final_embed_now, final_embed_now)
        time_weights_now = time_weights_now.sum(dim=1)
        weighted_now = torch.mean(final_embed_now*time_weights_now.unsqueeze(2), dim=1)
        
        final_embed = (weighted_bef + weighted_now)/2.0
        
        final_input = self.clf(final_embed)
        
        return final_embed, final_input



class ATT_NEW_2(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, char_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_NEW_2, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        self.char_hidden = char_hidden
        #########################################################################################################
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_now = nn.Sequential(*network)
        
        self.talk_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.talk_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk_bef = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk_now = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_now = nn.Sequential(*network)
        
        self.gaze_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.gaze_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze_bef = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze_now = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        #########################################################################################################
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_bef = nn.Sequential(*network)

        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_now = nn.Sequential(*network)
        
        self.me_gaze_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.me_gaze_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze_bef = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))
        
        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze_now = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))
        #########################################################################################################
        
        # network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        # self.character_input_bef = nn.Sequential(*network)
        
        # network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        # self.character_input_now = nn.Sequential(*network)
        
        # self.char_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        # self.char_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        # self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        # self.transformer_encoder_char_bef = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))
        
        # self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        # self.transformer_encoder_char_now = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))
        #########################################################################################################
        # overall_fea_size = (self.talk_hidden+self.gaze_hidden+self.me_gaze_hidden+self.char_hidden)
        overall_fea_size = self.talk_hidden+self.gaze_hidden+self.me_gaze_hidden
        
        self.mean_weight_bef = nn.MultiheadAttention(embed_dim=overall_fea_size, num_heads=2, batch_first=True)
        self.mean_weight_now = nn.MultiheadAttention(embed_dim=overall_fea_size, num_heads=2, batch_first=True)

        if out_dim == 1:
            network = [nn.Linear(overall_fea_size*2, out_dim), nn.Sigmoid()]
        else:
            network = [nn.Linear(overall_fea_size*2, out_dim)]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk_bef, candi_bef, right_bef, middle_bef, left_bef, char_bef,\
                      talk_now, candi_now, right_now, middle_now, left_now, char_now):
        #########################################################################################################
        # pdb.set_trace()
        talk_bef = self.talk_input_bef(talk_bef)
        talk_now = self.talk_input_now(talk_now)
        
        talk_embed_bef = talk_bef.transpose(0, 1)
        talk_embed_bef = self.transformer_encoder_talk_bef.forward(talk_embed_bef)
        talk_embed_bef = talk_embed_bef.transpose(0, 1)
        talk_embed_bef = talk_embed_bef*self.talk_alpha_bef

        talk_embed_now = talk_now.transpose(0, 1)
        talk_embed_now = self.transformer_encoder_talk_now.forward(talk_embed_now)
        talk_embed_now = talk_embed_now.transpose(0, 1)
        talk_embed_now = talk_embed_now*self.talk_alpha_now
        #########################################################################################################
   
        other_gaze_bef = torch.cat([right_bef, middle_bef, left_bef], dim = 2)
        other_gaze_now = torch.cat([right_now, middle_now, left_now], dim = 2)
        
        other_gaze_bef = self.gaze_input_bef(other_gaze_bef)
        other_gaze_now = self.gaze_input_now(other_gaze_now)
        
        other_gaze_embed_bef = other_gaze_bef.transpose(0, 1)
        other_gaze_embed_bef = self.transformer_encoder_gaze_bef.forward(other_gaze_embed_bef)
        other_gaze_embed_bef = other_gaze_embed_bef.transpose(0, 1)
        other_gaze_embed_bef = other_gaze_embed_bef*self.gaze_alpha_bef
        
        other_gaze_embed_now = other_gaze_now.transpose(0, 1)
        other_gaze_embed_now = self.transformer_encoder_gaze_now.forward(other_gaze_embed_now)
        other_gaze_embed_now = other_gaze_embed_now.transpose(0, 1)
        other_gaze_embed_now = other_gaze_embed_now*self.gaze_alpha_now
        #########################################################################################################
        
        me_gaze_bef = self.me_gaze_input_bef(candi_bef)
        me_gaze_now = self.me_gaze_input_now(candi_now)
        
        me_gaze_embed_bef = me_gaze_bef.transpose(0, 1)
        me_gaze_embed_bef = self.transformer_encoder_me_gaze_bef.forward(me_gaze_embed_bef)
        me_gaze_embed_bef = me_gaze_embed_bef.transpose(0, 1)
        me_gaze_embed_bef = me_gaze_embed_bef*self.me_gaze_alpha_bef
        
        me_gaze_embed_now = me_gaze_now.transpose(0, 1)
        me_gaze_embed_now = self.transformer_encoder_me_gaze_now.forward(me_gaze_embed_now)
        me_gaze_embed_now = me_gaze_embed_now.transpose(0, 1)
        me_gaze_embed_now = me_gaze_embed_now*self.me_gaze_alpha_now
        #########################################################################################################
        # pdb.set_trace()
        # char_gaze_bef = self.character_input_bef(char_bef)
        # char_gaze_now = self.character_input_now(char_now)
        
        # char_embed_bef = char_gaze_bef.transpose(0, 1)
        # char_embed_bef = self.transformer_encoder_char_bef.forward(char_embed_bef)
        # char_embed_bef = char_embed_bef.transpose(0, 1)
        # char_embed_bef = char_embed_bef*self.char_alpha_bef
        
        # char_embed_now = char_gaze_now.transpose(0, 1)
        # char_embed_now = self.transformer_encoder_char_now.forward(char_embed_now)
        # char_embed_now = char_embed_now.transpose(0, 1)
        # char_embed_now = char_embed_now*self.char_alpha_now
        #########################################################################################################
        
        # final_embed_bef = torch.cat([talk_embed_bef, other_gaze_embed_bef, me_gaze_embed_bef, char_embed_bef], dim = 2)
        # final_embed_now = torch.cat([talk_embed_now, other_gaze_embed_now, me_gaze_embed_now, char_embed_now], dim = 2)
        
        final_embed_bef = torch.cat([talk_embed_bef, other_gaze_embed_bef, me_gaze_embed_bef], dim = 2)
        final_embed_now = torch.cat([talk_embed_now, other_gaze_embed_now, me_gaze_embed_now], dim = 2)
        
        # pdb.set_trace()
        _, time_weights_bef = self.mean_weight_bef(final_embed_bef, final_embed_bef, final_embed_bef)
        time_weights_bef = time_weights_bef.sum(dim=1)
        weighted_bef = torch.mean(final_embed_bef*time_weights_bef.unsqueeze(2), dim=1)
        
        _, time_weights_now = self.mean_weight_now(final_embed_now, final_embed_now, final_embed_now)
        time_weights_now = time_weights_now.sum(dim=1)
        weighted_now = torch.mean(final_embed_now*time_weights_now.unsqueeze(2), dim=1)
        
        final_embed = torch.cat([weighted_bef, weighted_now], dim = 1)
        
        final_input = self.clf(final_embed)
        
        return final_embed, final_input

#%%


class ATT_NEW_3(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, char_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_NEW_3, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        self.char_hidden = char_hidden
        #########################################################################################################
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_now = nn.Sequential(*network)
        
        self.talk_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.talk_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk_bef = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk_now = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_now = nn.Sequential(*network)
        
        self.gaze_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.gaze_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze_bef = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze_now = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        #########################################################################################################
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_bef = nn.Sequential(*network)

        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_now = nn.Sequential(*network)
        
        self.me_gaze_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.me_gaze_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze_bef = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))
        
        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze_now = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_now = nn.Sequential(*network)
        
        self.char_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.char_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char_bef = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char_now = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))
        #########################################################################################################
        overall_fea_size = (self.talk_hidden+self.gaze_hidden+self.me_gaze_hidden+self.char_hidden)
        
        self.mean_weight_bef = nn.MultiheadAttention(embed_dim=overall_fea_size, num_heads=2, batch_first=True)
        self.mean_weight_now = nn.MultiheadAttention(embed_dim=overall_fea_size, num_heads=2, batch_first=True)

        if out_dim == 1:
            network = [nn.Linear(overall_fea_size*2, out_dim), nn.Sigmoid()]
        else:
            network = [nn.Linear(overall_fea_size*2, out_dim)]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk_bef, candi_bef, right_bef, middle_bef, left_bef, char_bef,\
                      talk_now, candi_now, right_now, middle_now, left_now, char_now):
        #########################################################################################################
        # pdb.set_trace()
        talk_bef = self.talk_input_bef(talk_bef)
        talk_now = self.talk_input_now(talk_now)
        
        talk_embed_bef = talk_bef.transpose(0, 1)
        talk_embed_bef = self.transformer_encoder_talk_bef.forward(talk_embed_bef)
        talk_embed_bef = talk_embed_bef.transpose(0, 1)
        talk_embed_bef = talk_embed_bef*self.talk_alpha_bef

        talk_embed_now = talk_now.transpose(0, 1)
        talk_embed_now = self.transformer_encoder_talk_now.forward(talk_embed_now)
        talk_embed_now = talk_embed_now.transpose(0, 1)
        talk_embed_now = talk_embed_now*self.talk_alpha_now
        #########################################################################################################
   
        other_gaze_bef = torch.cat([right_bef, middle_bef, left_bef], dim = 2)
        other_gaze_now = torch.cat([right_now, middle_now, left_now], dim = 2)
        
        other_gaze_bef = self.gaze_input_bef(other_gaze_bef)
        other_gaze_now = self.gaze_input_now(other_gaze_now)
        
        other_gaze_embed_bef = other_gaze_bef.transpose(0, 1)
        other_gaze_embed_bef = self.transformer_encoder_gaze_bef.forward(other_gaze_embed_bef)
        other_gaze_embed_bef = other_gaze_embed_bef.transpose(0, 1)
        other_gaze_embed_bef = other_gaze_embed_bef*self.gaze_alpha_bef
        
        other_gaze_embed_now = other_gaze_now.transpose(0, 1)
        other_gaze_embed_now = self.transformer_encoder_gaze_now.forward(other_gaze_embed_now)
        other_gaze_embed_now = other_gaze_embed_now.transpose(0, 1)
        other_gaze_embed_now = other_gaze_embed_now*self.gaze_alpha_now
        #########################################################################################################
        
        me_gaze_bef = self.me_gaze_input_bef(candi_bef)
        me_gaze_now = self.me_gaze_input_now(candi_now)
        
        me_gaze_embed_bef = me_gaze_bef.transpose(0, 1)
        me_gaze_embed_bef = self.transformer_encoder_me_gaze_bef.forward(me_gaze_embed_bef)
        me_gaze_embed_bef = me_gaze_embed_bef.transpose(0, 1)
        me_gaze_embed_bef = me_gaze_embed_bef*self.me_gaze_alpha_bef
        
        me_gaze_embed_now = me_gaze_now.transpose(0, 1)
        me_gaze_embed_now = self.transformer_encoder_me_gaze_now.forward(me_gaze_embed_now)
        me_gaze_embed_now = me_gaze_embed_now.transpose(0, 1)
        me_gaze_embed_now = me_gaze_embed_now*self.me_gaze_alpha_now
        #########################################################################################################
        # pdb.set_trace()
        char_gaze_bef = self.character_input_bef(char_bef)
        char_gaze_now = self.character_input_now(char_now)
        
        char_embed_bef = char_gaze_bef.transpose(0, 1)
        char_embed_bef = self.transformer_encoder_char_bef.forward(char_embed_bef)
        char_embed_bef = char_embed_bef.transpose(0, 1)
        char_embed_bef = char_embed_bef*self.char_alpha_bef
        
        char_embed_now = char_gaze_now.transpose(0, 1)
        char_embed_now = self.transformer_encoder_char_now.forward(char_embed_now)
        char_embed_now = char_embed_now.transpose(0, 1)
        char_embed_now = char_embed_now*self.char_alpha_now
        #########################################################################################################
        
        final_embed_bef = torch.cat([talk_embed_bef, other_gaze_embed_bef, me_gaze_embed_bef, char_embed_bef], dim = 2)
        final_embed_now = torch.cat([talk_embed_now, other_gaze_embed_now, me_gaze_embed_now, char_embed_now], dim = 2)
        
        
        # pdb.set_trace()
        _, time_weights_bef = self.mean_weight_bef(final_embed_bef, final_embed_bef, final_embed_bef)
        time_weights_bef = time_weights_bef.sum(dim=1)
        weighted_bef = torch.mean(final_embed_bef*time_weights_bef.unsqueeze(2), dim=1)
        
        _, time_weights_now = self.mean_weight_now(final_embed_now, final_embed_now, final_embed_now)
        time_weights_now = time_weights_now.sum(dim=1)
        weighted_now = torch.mean(final_embed_now*time_weights_now.unsqueeze(2), dim=1)
        
        final_embed = torch.cat([weighted_bef, weighted_now], dim = 1)
        
        final_input = self.clf(final_embed)
        
        return final_embed, final_input

#%%


class ATT_NEW_4(nn.Module):
    def __init__(self, talk_fea, talk_hidden, me_gaze_fea, me_gaze_hidden, gaze_fea, gaze_hidden, char_hidden, layer_num, tfm_head_talk, tfm_head_gaze, head_me_gaze, out_dim):
        super(ATT_NEW_4, self).__init__()
        
        self.layers = layer_num
        self.head_talk = tfm_head_talk
        self.head_gaze = tfm_head_gaze
        self.head_me_gaze = head_me_gaze
        self.talk_hidden = talk_hidden
        self.gaze_hidden = gaze_hidden
        self.me_gaze_hidden = me_gaze_hidden
        self.char_hidden = char_hidden
        #########################################################################################################
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(talk_fea, talk_hidden), nn.ReLU()]
        self.talk_input_now = nn.Sequential(*network)
        
        self.talk_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.talk_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk_bef = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        
        self.encoder_layer_talk = nn.TransformerEncoderLayer(d_model=self.talk_hidden, dim_feedforward=self.talk_hidden, nhead=self.head_talk, activation='gelu', dropout = 0.2)
        self.transformer_encoder_talk_now = nn.TransformerEncoder(self.encoder_layer_talk, num_layers=self.layers, norm=nn.LayerNorm(self.talk_hidden))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea*3, gaze_hidden), nn.ReLU()]
        self.gaze_input_now = nn.Sequential(*network)
        
        self.gaze_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.gaze_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze_bef = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        
        self.encoder_layer_gaze = nn.TransformerEncoderLayer(d_model=self.gaze_hidden, dim_feedforward=self.gaze_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_gaze_now = nn.TransformerEncoder(self.encoder_layer_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.gaze_hidden))
        #########################################################################################################
        
        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_bef = nn.Sequential(*network)

        network = [nn.Linear(me_gaze_fea, self.me_gaze_hidden), nn.ReLU()]
        self.me_gaze_input_now = nn.Sequential(*network)
        
        self.me_gaze_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.me_gaze_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze_bef = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))
        
        self.encoder_layer_me_gaze = nn.TransformerEncoderLayer(d_model=self.me_gaze_hidden, dim_feedforward=self.me_gaze_hidden, nhead=self.head_me_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_me_gaze_now = nn.TransformerEncoder(self.encoder_layer_me_gaze, num_layers=self.layers, norm=nn.LayerNorm(self.me_gaze_hidden))
        #########################################################################################################
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_bef = nn.Sequential(*network)
        
        network = [nn.Linear(gaze_fea, char_hidden), nn.ReLU()]
        self.character_input_now = nn.Sequential(*network)
        
        self.char_alpha_bef = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.char_alpha_now = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char_bef = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))
        
        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=self.char_hidden, dim_feedforward=self.char_hidden, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_char_now = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(self.char_hidden))
        #########################################################################################################
        overall_fea_size = (self.talk_hidden+self.gaze_hidden+self.me_gaze_hidden+self.char_hidden)
        
        self.mean_weight_bef = nn.MultiheadAttention(embed_dim=overall_fea_size, num_heads=2, batch_first=True)
        self.mean_weight_now = nn.MultiheadAttention(embed_dim=overall_fea_size, num_heads=2, batch_first=True)

        self.encoder_layer_char = nn.TransformerEncoderLayer(d_model=overall_fea_size*2, dim_feedforward=overall_fea_size*2, nhead=self.head_gaze, activation='gelu', dropout = 0.2)
        self.transformer_encoder_final = nn.TransformerEncoder(self.encoder_layer_char, num_layers=self.layers, norm=nn.LayerNorm(overall_fea_size*2))


        if out_dim == 1:
            network = [nn.Linear(overall_fea_size*2, out_dim), nn.Sigmoid()]
        else:
            network = [nn.Linear(overall_fea_size*2, out_dim)]
        self.clf = nn.Sequential(*network)
        
    def forward(self, talk_bef, candi_bef, right_bef, middle_bef, left_bef, char_bef,\
                      talk_now, candi_now, right_now, middle_now, left_now, char_now):
        #########################################################################################################
        # pdb.set_trace()
        talk_bef = self.talk_input_bef(talk_bef)
        talk_now = self.talk_input_now(talk_now)
        
        talk_embed_bef = talk_bef.transpose(0, 1)
        talk_embed_bef = self.transformer_encoder_talk_bef.forward(talk_embed_bef)
        talk_embed_bef = talk_embed_bef.transpose(0, 1)
        talk_embed_bef = talk_embed_bef*self.talk_alpha_bef

        talk_embed_now = talk_now.transpose(0, 1)
        talk_embed_now = self.transformer_encoder_talk_now.forward(talk_embed_now)
        talk_embed_now = talk_embed_now.transpose(0, 1)
        talk_embed_now = talk_embed_now*self.talk_alpha_now
        #########################################################################################################
   
        other_gaze_bef = torch.cat([right_bef, middle_bef, left_bef], dim = 2)
        other_gaze_now = torch.cat([right_now, middle_now, left_now], dim = 2)
        
        other_gaze_bef = self.gaze_input_bef(other_gaze_bef)
        other_gaze_now = self.gaze_input_now(other_gaze_now)
        
        other_gaze_embed_bef = other_gaze_bef.transpose(0, 1)
        other_gaze_embed_bef = self.transformer_encoder_gaze_bef.forward(other_gaze_embed_bef)
        other_gaze_embed_bef = other_gaze_embed_bef.transpose(0, 1)
        other_gaze_embed_bef = other_gaze_embed_bef*self.gaze_alpha_bef
        
        other_gaze_embed_now = other_gaze_now.transpose(0, 1)
        other_gaze_embed_now = self.transformer_encoder_gaze_now.forward(other_gaze_embed_now)
        other_gaze_embed_now = other_gaze_embed_now.transpose(0, 1)
        other_gaze_embed_now = other_gaze_embed_now*self.gaze_alpha_now
        #########################################################################################################
        
        me_gaze_bef = self.me_gaze_input_bef(candi_bef)
        me_gaze_now = self.me_gaze_input_now(candi_now)
        
        me_gaze_embed_bef = me_gaze_bef.transpose(0, 1)
        me_gaze_embed_bef = self.transformer_encoder_me_gaze_bef.forward(me_gaze_embed_bef)
        me_gaze_embed_bef = me_gaze_embed_bef.transpose(0, 1)
        me_gaze_embed_bef = me_gaze_embed_bef*self.me_gaze_alpha_bef
        
        me_gaze_embed_now = me_gaze_now.transpose(0, 1)
        me_gaze_embed_now = self.transformer_encoder_me_gaze_now.forward(me_gaze_embed_now)
        me_gaze_embed_now = me_gaze_embed_now.transpose(0, 1)
        me_gaze_embed_now = me_gaze_embed_now*self.me_gaze_alpha_now
        #########################################################################################################
        # pdb.set_trace()
        char_gaze_bef = self.character_input_bef(char_bef)
        char_gaze_now = self.character_input_now(char_now)
        
        char_embed_bef = char_gaze_bef.transpose(0, 1)
        char_embed_bef = self.transformer_encoder_char_bef.forward(char_embed_bef)
        char_embed_bef = char_embed_bef.transpose(0, 1)
        char_embed_bef = char_embed_bef*self.char_alpha_bef
        
        char_embed_now = char_gaze_now.transpose(0, 1)
        char_embed_now = self.transformer_encoder_char_now.forward(char_embed_now)
        char_embed_now = char_embed_now.transpose(0, 1)
        char_embed_now = char_embed_now*self.char_alpha_now
        #########################################################################################################
        
        final_embed_bef = torch.cat([talk_embed_bef, other_gaze_embed_bef, me_gaze_embed_bef, char_embed_bef], dim = 2)
        final_embed_now = torch.cat([talk_embed_now, other_gaze_embed_now, me_gaze_embed_now, char_embed_now], dim = 2)
        
        # pdb.set_trace()
        _, time_weights_bef = self.mean_weight_bef(final_embed_bef, final_embed_bef, final_embed_bef)
        time_weights_bef = time_weights_bef.sum(dim=1)
        weighted_bef = final_embed_bef*time_weights_bef.unsqueeze(2)
        
        _, time_weights_now = self.mean_weight_now(final_embed_now, final_embed_now, final_embed_now)
        time_weights_now = time_weights_now.sum(dim=1)
        weighted_now = final_embed_now*time_weights_now.unsqueeze(2)
        
        final_embed = torch.cat([weighted_bef, weighted_now], dim = 2)

        final_embed = final_embed.transpose(0, 1)
        final_embed = self.transformer_encoder_final.forward(final_embed)
        final_embed = final_embed.transpose(0, 1)
        
        final_embed = torch.mean(final_embed, dim = 1)

        final_input = self.clf(final_embed)
        
        return final_embed, final_input

#%%
if __name__=='__main__':
    
    device = torch.device( "cpu")
    
    
    batch_size = 16
    time_step = 30
    talk_bef = torch.rand(batch_size, time_step, 1).to(device)
    talk_now = torch.rand(batch_size, time_step, 1).to(device)
    me_gaze_bef = torch.rand(batch_size, time_step, 4).to(device)
    me_gaze_now = torch.rand(batch_size, time_step, 4).to(device)
    other_bef = torch.rand(batch_size, time_step, 1).to(device)
    other_now = torch.rand(batch_size, time_step, 1).to(device)
    char_bef = torch.rand(batch_size, time_step, 1).to(device)
    char_now = torch.rand(batch_size, time_step, 1).to(device)
    # pdb.set_trace()
    # (self, fea_dim, tfm_head, out_dim)
    model = ATT_NEW_5(talk_fea=1, talk_hidden=8, me_gaze_fea = 4, me_gaze_hidden = 16, gaze_fea = 1, gaze_hidden = 16, char_hidden=8,
                           layer_num = 1, tfm_head_talk = 2, tfm_head_gaze = 8, head_me_gaze = 2, out_dim = 4).to(device)
    
    _, output_trans = model.forward(talk_bef,
                                    me_gaze_bef.to(device),other_bef.float().to(device),
                                    other_bef.float().to(device),other_bef.float().to(device),char_bef.float().to(device),
                                    talk_now,
                                    me_gaze_bef.float().to(device),other_bef.float().to(device),char_now.float().to(device),
                                    other_bef.float().to(device),other_bef.float().to(device))
    
    