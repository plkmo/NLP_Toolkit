#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 09:22:51 2020

@author: tsd
"""
import os
import pickle
import torch

class Config():
    data_path = './data/yelp/'
    log_dir = 'runs/exp'
    save_path = './data/style_transfer'
    pretrained_embed_path = './embedding/'
    device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi' # 'Multi' or 'Cond'
    load_pretrained_embed = False
    min_freq = 3
    max_length = 16
    embed_size = 256
    d_model = 256
    h = 4
    num_styles = 2
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0
    
    def __init__(self, args):
        self.data_path = args.data_path
        self.num_styles = args.num_classes
        self.batch_size = args.batch_size
        self.max_length = args.max_features_length
        self.d_model = args.d_model
        self.embed_size = args.d_model
        self.h = args.n_heads
        self.lr_D = args.lr_D
        self.lr_F = args.lr_F
        self.num_layers = args.num
        self.num_iters = args.num_iters
        self.checkpoint_Fpath = args.checkpoint_Fpath
        self.checkpoint_Fpath = args.checkpoint_Fpath
        self.eval_steps = args.save_iters

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)