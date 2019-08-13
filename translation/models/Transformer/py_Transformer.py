# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:01:15 2019

@author: WT
"""

import torch
import torch.nn as nn

### create masks for src & trg sequences
def create_masks(src, trg):
    src_mask = (src == 1).unsqueeze(-2).bool()
    if trg is not None:
        trg_mask = (trg == 1).unsqueeze(-2).bool()
    else:
        trg_mask = None
    src_mask = src_mask[:,0,:]
    trg_mask = trg_mask[:,0,:]
    return src_mask, trg_mask

def create_trg_mask(trg, cuda):
    trg_mask = (trg == 1).unsqueeze(-2).bool()
    trg_mask = trg_mask[:,0,:]
    return trg_mask

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
        self.embed1 = nn.Embedding(src_vocab, d_model)
        self.embed2 = nn.Embedding(trg_vocab, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=num,\
                                          num_decoder_layers=num, dim_feedforward=ff_dim, dropout=0.1)
        self.fc1 = nn.Linear(d_model, trg_vocab)
    
    def forward(self, src, trg, src_mask, trg_mask=None, infer=False, trg_vocab_obj=None):
        #print(src[0,:], trg[0,:])
        src = self.embed1(src)
        trg = self.embed2(trg)
        src = src.permute(1,0,2)
        trg = trg.permute(1,0,2)
        out = self.transformer(src, trg, src_key_padding_mask=src_mask, \
                               tgt_key_padding_mask=trg_mask)
        out = out.permute(1,0,2)
        out = self.fc1(out)
        #print(out.shape)
        #print(out[0,:,:])
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