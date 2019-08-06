# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:22:31 2019

@author: WT
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import copy

### create masks for src & trg sequences
def create_masks(src, trg):
    src_mask = (src != 1).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != 1).unsqueeze(-2)
        np_mask = np.triu(np.ones((1, trg.size(1),trg.size(1))),k=1).astype('uint8')
        np_mask = Variable(torch.from_numpy(np_mask) == 0)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask

def create_trg_mask(trg, cuda):
    trg_mask = (trg != 1).unsqueeze(-2)
    np_mask = np.triu(np.ones((1, trg.size(1),trg.size(1))),k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if cuda:
        np_mask = np_mask.cuda()
    trg_mask = trg_mask & np_mask
    return trg_mask

class Pos_Encoder(nn.Module):
    def __init__(self, d_model, max_len):
        super(Pos_Encoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                #pe[pos, i] = math.sin(pos/(10000**((2*i)/d_model)))
                #pe[pos, i + 1] = math.cos(pos/(10000**((2*(i + 1))/d_model)))
                pe[pos, i] = math.sin(pos/(10000**((i)/d_model)))
                pe[pos, i + 1] = math.cos(pos/(10000**(((i))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # input = batch_size X seq_len X d_model
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x
    
def Attention(q, k, v, dh, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(dh)
    if mask is not None:
        mask = mask.unsqueeze(1); #print("Mask", mask.shape); print("scores", scores.shape)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = torch.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MHAttention(nn.Module):
    def __init__(self, d_model, n_heads, droprate=0.1):
        super(MHAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model//n_heads
        # learning layers for q,k,v
        self.q_matrix = nn.Linear(d_model, d_model)
        self.k_matrix = nn.Linear(d_model, d_model)
        self.v_matrix = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(droprate)
        self.fc1 = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        # input = batch_size X seq_len X d_model into batch_size X heads X seq_len X d_model/heads
        q = self.q_matrix(q); q = q.view(q.size(0), self.n_heads, -1, self.dh)
        k = self.k_matrix(k); k = k.view(k.size(0), self.n_heads, -1, self.dh)
        v = self.v_matrix(v); v = v.view(v.size(0), self.n_heads, -1, self.dh)
        scores = Attention(q, k, v, self.dh, mask, self.dropout)
        scores = scores.reshape(q.size(0), -1, self.d_model)
        output = self.fc1(scores)
        return output   

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_size=2048, droprate=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.dropout = nn.Dropout(droprate)
        self.fc2 = nn.Linear(hidden_size, d_model)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.alpha = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))
    
    def forward(self, x):
        norm = self.alpha*(x - x.mean(dim=-1, keepdim=True))/(x.std(dim=-1, keepdim=True) + 1e-7) + \
                            self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, droprate=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MHAttention(d_model=d_model, n_heads=n_heads)
        self.dropout1 = nn.Dropout(droprate)
        self.norm2 = LayerNorm(d_model)
        self.fc1 = FeedForward(d_model=d_model)    
        self.dropout2 = nn.Dropout(droprate)
    
    def forward(self, x, mask):
        x1 = self.norm1(x); #print("e1", x1.shape)
        x = x + self.dropout1(self.attn(x1, x1, x1, mask)); #print("e2", x.shape)
        x1 = self.norm2(x)
        x = x + self.dropout2(self.fc1(x1)); #print("e3", x.shape)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, droprate=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(droprate)
        self.dropout2 = nn.Dropout(droprate)
        self.dropout3 = nn.Dropout(droprate)
        self.attn1 = MHAttention(d_model=d_model, n_heads=n_heads)
        self.attn2 = MHAttention(d_model=d_model, n_heads=n_heads)
        self.fc1 = FeedForward(d_model=d_model)
        
    def forward(self, x, e_out, src_mask, trg_mask):
        x1 = self.norm1(x); #print("d1", x1.shape)
        x = x + self.dropout1(self.attn1(x1, x1, x1, trg_mask)); #print("d2", x.shape)
        x1 = self.norm2(x)
        x = x + self.dropout2(self.attn2(x1, e_out, e_out, src_mask)); #print("d3", x.shape)
        x1 = self.norm3(x)
        x = x + self.dropout3(self.fc1(x1)); #print("d4", x.shape)
        return x

def clone_layers(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])    
    
class EncoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model, num, n_heads):
        super(EncoderBlock, self).__init__()
        self.num = num
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = Pos_Encoder(d_model, max_len= 80)
        self.layers = clone_layers(EncoderLayer(d_model, n_heads), num)
        self.norm = LayerNorm(d_model)
    
    def forward(self, src, mask):
        x = self.embed(src); #print("e_embed", x.shape)
        x = self.pe(x)
        for i in range(self.num):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model, num, n_heads):
        super(DecoderBlock, self).__init__()
        self.num = num
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = Pos_Encoder(d_model, max_len = 80)
        self.layers = clone_layers(DecoderLayer(d_model, n_heads), num)
        self.norm = LayerNorm(d_model)
    
    def forward(self, trg, e_out, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.num):
            x = self.layers[i](x, e_out, src_mask, trg_mask)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, num, n_heads):
        super(Transformer, self).__init__()
        self.encoder = EncoderBlock(vocab_size=src_vocab, d_model=d_model, num=num, n_heads=n_heads)
        self.decoder = DecoderBlock(vocab_size=trg_vocab, d_model=d_model, num=num, n_heads=n_heads)
        self.fc1 = nn.Linear(d_model, trg_vocab)
    
    def forward(self, src, trg, src_mask, trg_mask):
        e_out = self.encoder(src, src_mask); #print("e_out", e_out.shape)
        d_out = self.decoder(trg, e_out, src_mask, trg_mask); #print("d_out", d_out.shape)
        x = self.fc1(d_out); #print("x", x.shape)
        return x