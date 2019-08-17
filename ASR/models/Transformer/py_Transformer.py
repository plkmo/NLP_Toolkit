# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:01:15 2019

@author: WT
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def create_window_mask(size, window_len=10):
    m = np.zeros((size, size), dtype=float)
    for j in range(len(m)):
        for k in range(len(m)):
            if abs(j-k) > window_len:
                m[j, k] = float('-inf') 
    m = Variable(torch.from_numpy(m)).float()
    #m = m.bool(); #print(m)
    return m

### create masks for src & trg sequences
def create_masks(src, trg, f_len, args):
    init_len = args.max_frame_len
    if args.max_frame_len % 4 == 0:
        final_len = int(args.max_frame_len/4)
    else:
        final_len = int(args.max_frame_len/4) + 1 #793
    src_mask = torch.zeros((src.shape[0], final_len)).long()
    for i in range(len(src[:,0,0,0])):
        src_ratio = f_len[i].item()/init_len
        src_mask[i, int(src_ratio*final_len):] = 1
    src_mask = src_mask.bool()
    
    if trg is not None:
        trg_mask = (trg == 1).unsqueeze(-2).bool()
    else:
        trg_mask = None
    trg_mask = trg_mask[:,0,:]
    return src_mask, trg_mask

def create_trg_mask(trg, cuda):
    trg_mask = (trg == 1).unsqueeze(-2).bool()
    trg_mask = trg_mask[:,0,:]
    return trg_mask

class Conv2dBlock(nn.Module):
    def __init__(self, c_in, c_out=64):
        super(Conv2dBlock, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.drop1 = nn.Dropout(p=0.1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x))); #print(x.shape)
        x = self.drop1(x)
        x = torch.relu(self.bn2(self.conv2(x))); #print(x.shape)
        return x

class pyTransformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, ff_dim, num, n_heads,\
                 max_encoder_len=80, max_decoder_len=80):
        super(pyTransformer, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num = num
        self.n_heads = n_heads
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.conv = Conv2dBlock(3, 64)
        self.embed1 = nn.Linear(int(64*src_vocab/12), d_model)
        self.embed2 = nn.Embedding(trg_vocab, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=num,\
                                          num_decoder_layers=num, dim_feedforward=ff_dim, dropout=0.1)
        self.fc1 = nn.Linear(d_model, trg_vocab)
    
    def forward(self, src, trg, src_mask, trg_mask=None, g_mask1=None, g_mask2=None, \
                infer=False, trg_vocab_obj=None):
        src = self.conv(src) 
        #print(src)
        src = src.reshape(src.shape[0], src.shape[-1], -1)
        src = self.embed1(src)
        trg = self.embed2(trg)
        src = src.permute(1,0,2)
        trg = trg.permute(1,0,2)
        if not infer:
            if g_mask1 is not None:
                #print(g_mask1)
                #print(src)
                out = self.transformer(src, trg, src_mask=g_mask1, src_key_padding_mask=src_mask, \
                                   tgt_key_padding_mask=trg_mask)
            else:
                out = self.transformer(src, trg, src_key_padding_mask=src_mask, \
                                   tgt_key_padding_mask=trg_mask)
            out = out.permute(1,0,2)
            out = self.fc1(out)
            #print(out.shape)
            #print(out[0,:,:])
            #print(self.conv.conv1.weight.grad)
        else:
            pass
        return out
    
    @classmethod # src_vocab, trg_vocab, d_model, num, n_heads
    def load_model(cls, path):
        checkpoint = torch.load(path)
        model = cls(src_vocab=checkpoint["src_vocab"], \
                    trg_vocab=checkpoint["trg_vocab"], \
                    d_model=checkpoint["d_model"], \
                    ff_dim=checkpoint["ff_dim"], \
                    num=checkpoint["num"], \
                    n_heads=checkpoint["n_heads"], \
                    max_encoder_len=checkpoint["max_encoder_len"], \
                    max_decoder_len=checkpoint["max_decoder_len"], \
                    )
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def save_state(self, epoch, optimizer, scheduler, best_acc, path):
        state = {
                    'epoch': epoch + 1,\
                    'state_dict': self.state_dict(),\
                    'best_acc': best_acc,\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'src_vocab' : self.src_vocab,\
                    'trg_vocab': self.trg_vocab,\
                    'd_model': self.d_model,\
                    'ff_dim': self.ff_dim,\
                    'num': self.num,\
                    'n_heads': self.n_heads,\
                    'max_encoder_len': self.max_encoder_len,\
                    'max_decoder_len': self.max_decoder_len,
                }
        torch.save(state, path)