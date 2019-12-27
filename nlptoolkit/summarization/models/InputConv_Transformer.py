# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:09:15 2019

@author: WT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math
import copy


def create_window_mask(size, window_len=10):
    m = np.zeros((size, size))
    for j in range(len(m)):
        for k in range(len(m)):
            if abs(j-k) > window_len:
                m[j, k] = -1e9
    m = Variable(torch.from_numpy(m))
    return m

def create_gaussian_mask(size):
    m = np.ones((size, size))
    for j in range(len(m)):
        for k in range(len(m)):
            m[j, k] = (-(j-k)**2)
    m = m/2
    m = Variable(torch.from_numpy(m))
    return m
    
### create masks for src & trg sequences
def create_masks(src, trg):
    init_len = src.shape[-1]
    if init_len % 8 == 0:
        final_len = int(init_len/8)
    else:
        final_len = int(init_len/8) + 1
    src_mask = torch.ones((src.shape[0], 1, final_len)).long()
    for i in range(len(src[:,0])):
        src_ratio = (init_len - (src[i] == 1).sum().item())/init_len
        src_mask[i, :, int(src_ratio*final_len):] = 0
    #src_mask = (src != 0).float().mean(dim=2).long().unsqueeze(1)
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
            '''
            for i in range(0, d_model, 2):
            
                #pe[pos, i] = math.sin(pos/(10000**((2*i)/d_model)))
                #pe[pos, i + 1] = math.cos(pos/(10000**((2*(i + 1))/d_model)))
                pe[pos, i] = math.sin(pos/(10000**((i)/d_model)))
                pe[pos, i + 1] = math.cos(pos/(10000**(((i))/d_model)))
            '''
            
            for i in range(0, d_model, 1):
                if i < d_model/2:
                    pe[pos, i] = math.sin(pos/(10000**((2*i)/d_model)))
                else:
                    pe[pos, i] = math.cos(pos/(10000**((2*i)/d_model)))
            
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # input = batch_size X seq_len X d_model
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x
    
def Attention(q, k, v, dh, mask=None, g_mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(dh)
    if mask is not None:
        mask = mask.unsqueeze(1); #print("Mask", mask.shape); print("scores", scores.shape)
        scores = scores.masked_fill(mask == 0, -5e4)
    
    if g_mask is not None:
        scores = scores + g_mask
    
    scores = torch.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MHAttention(nn.Module):
    def __init__(self, d_model, n_heads, gaussian_mask=False, src_size=0, droprate=0.01):
        super(MHAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model//n_heads
        self.gaussian_mask = gaussian_mask
        #if self.gaussian_mask:
            #for head in range(self.n_heads):
            #    setattr(self, "sigma_%i" % head, nn.parameter.Parameter(torch.FloatTensor(1).normal_()))
        # learning layers for q,k,v
        self.q_matrix = nn.Linear(d_model, d_model)
        self.k_matrix = nn.Linear(d_model, d_model)
        self.v_matrix = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(droprate)
        self.fc1 = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None, g_mask=None):
        # input = batch_size X seq_len X d_model into batch_size X heads X seq_len X d_model/heads
        q = self.q_matrix(q); q = q.view(q.size(0), self.n_heads, -1, self.dh); #print("q", q.shape)
        k = self.k_matrix(k); k = k.view(k.size(0), self.n_heads, -1, self.dh); #print("k", k.shape)
        v = self.v_matrix(v); v = v.view(v.size(0), self.n_heads, -1, self.dh); #print("v", v.shape)
        if self.gaussian_mask:
            #g_mask = torch.cat([g_mask.unsqueeze(0).unsqueeze(0)/(15*getattr(self, "sigma_%i" % head))**2 \
            #                    for head in range(self.n_heads)], dim=1); #print("g_mask", g_mask.shape)
            g_mask = torch.cat([g_mask.unsqueeze(0).unsqueeze(0) \
                                for head in range(self.n_heads)], dim=1);
        scores = Attention(q, k, v, self.dh, mask, g_mask=g_mask, dropout=self.dropout); #print("scores", scores.shape)
        scores = scores.reshape(q.size(0), -1, self.d_model)
        output = self.fc1(scores)
        return output   

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_size=1024, droprate=0.01):
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
    def __init__(self, d_model, n_heads, ff_dim, droprate=0.01):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MHAttention(d_model=d_model, n_heads=n_heads, gaussian_mask=False)
        self.dropout1 = nn.Dropout(droprate)
        self.norm2 = LayerNorm(d_model)
        self.fc1 = FeedForward(d_model=d_model, hidden_size=ff_dim)    
        self.dropout2 = nn.Dropout(droprate)
    
    def forward(self, x, mask, g_mask):
        x1 = self.norm1(x); #print("e1", x1.shape)
        x = x + self.dropout1(self.attn(x1, x1, x1, mask, g_mask)); #print("e2", x.shape)
        x1 = self.norm2(x)
        x = x + self.dropout2(self.fc1(x1)); #print("e3", x.shape)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, droprate=0.01):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(droprate)
        self.dropout2 = nn.Dropout(droprate)
        self.dropout3 = nn.Dropout(droprate)
        self.attn1 = MHAttention(d_model=d_model, n_heads=n_heads, gaussian_mask=False)
        self.attn2 = MHAttention(d_model=d_model, n_heads=n_heads, gaussian_mask=False)
        self.fc1 = FeedForward(d_model=d_model, hidden_size=ff_dim)
        
    def forward(self, x, e_out, src_mask, trg_mask, g_mask2):
        x1 = self.norm1(x); #print("d1", x1.shape)
        x = x + self.dropout1(self.attn1(x1, x1, x1, trg_mask, g_mask2)); #print("d2", x.shape)
        x1 = self.norm2(x)
        x = x + self.dropout2(self.attn2(x1, e_out, e_out, src_mask)); #print("d3", x.shape)
        x1 = self.norm3(x)
        x = x + self.dropout3(self.fc1(x1)); #print("d4", x.shape)
        return x

def clone_layers(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])    

class Conv1dBlock(nn.Module):
    def __init__(self, c_in, c_out=64):
        super(Conv1dBlock, self).__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv1d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.bn3 = nn.BatchNorm1d(c_out)
        self.drop1 = nn.Dropout(p=0.05)
        self.drop2 = nn.Dropout(p=0.05)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x))); #print(x.shape)
        x = self.drop1(x)
        x = torch.relu(self.bn2(self.conv2(x))); #print(x.shape)
        x = self.drop2(x)
        x = torch.relu(self.bn3(self.conv3(x))); #print(x.shape)
        return x
    
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
    
class EncoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model, ff_dim, num, n_heads, max_len):
        super(EncoderBlock, self).__init__()
        self.num = num
        self.embed = nn.Embedding(vocab_size, d_model)
        #self.norm = LayerNorm(d_model)
        self.pe = Pos_Encoder(d_model, max_len = max_len)
        self.conv = Conv1dBlock(d_model, d_model) # embed_dim = input channels
        self.layers = clone_layers(EncoderLayer(d_model, n_heads, ff_dim), num)
        self.norm1 = LayerNorm(d_model)
    
    def forward(self, src, mask, g_mask):
        x = self.embed(src); #print("e_embed", x.shape) # batch X seq_len X embed_dim
        x = self.pe(x)
        x = x.permute((0,2,1))
        x = self.conv(x)
        x = x.permute((0,2,1))
        for i in range(self.num):
            x = self.layers[i](x, mask, g_mask)
        x = self.norm1(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model, ff_dim, num, n_heads, max_len):
        super(DecoderBlock, self).__init__()
        self.num = num
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = Pos_Encoder(d_model, max_len=max_len)
        self.layers = clone_layers(DecoderLayer(d_model, n_heads, ff_dim), num)
        self.norm = LayerNorm(d_model)
    
    def forward(self, trg, e_out, src_mask, trg_mask, g_mask2):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.num):
            x = self.layers[i](x, e_out, src_mask, trg_mask, g_mask2)
        x = self.norm(x)
        return x

class SummaryTransformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, ff_dim, num, n_heads, max_encoder_len, max_decoder_len, use_conv=True):
        super(SummaryTransformer, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num = num
        self.n_heads = n_heads
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.encoder = EncoderBlock(vocab_size=src_vocab, d_model=d_model, ff_dim=ff_dim,\
                                    num=num, n_heads=n_heads, max_len=max_encoder_len)
        self.decoder = DecoderBlock(vocab_size=trg_vocab, d_model=d_model, ff_dim=ff_dim,\
                                    num=num, n_heads=n_heads, max_len=max_decoder_len)
        self.fc1 = nn.Linear(d_model, trg_vocab)
    
    def forward(self, src, trg, src_mask, trg_mask=None, g_mask1=None, g_mask2=None, infer=False):
        '''Runs a forward pass if infer=False.
        If infer=True (evaluation mode), generate text sequence given a sequence of features and returns the generated sequence indexes'''
        ### src = batch_size X seq_len
        e_out = self.encoder(src, src_mask, g_mask1); #print("e_out", e_out.shape)
        if not infer:
            d_out = self.decoder(trg, e_out, src_mask, trg_mask, g_mask2); #print("d_out", d_out.shape)
            x = self.fc1(d_out); #print("x", x.shape)
            return x
        else:
            o = torch.tensor([[]]).long()
            if src.is_cuda:
                o = o.cuda()
            for i in range(self.max_decoder_len):
                trg_mask = create_trg_mask(trg, src.is_cuda)
                d_out = self.decoder(trg, e_out, src_mask, trg_mask, g_mask2)
                x = self.fc1(d_out)
                o_labels = torch.softmax(x, dim=2).max(2)[1]
                trg = torch.cat((trg, o_labels[:,-1:]), dim=1)
                o = torch.cat((o, o_labels[:,-1:]), dim=1)
                if o_labels[0, -1].item() == 2: # break if <eos> token encountered
                    break
            return trg
            #return o
    
    @classmethod
    def load_model(cls, path, args, cuda=True, amp=None):
        checkpoint = torch.load(path)
        model = cls(src_vocab=checkpoint["src_vocab"], \
                    trg_vocab=checkpoint["trg_vocab"], \
                    d_model=checkpoint["d_model"], \
                    ff_dim=checkpoint["ff_dim"], \
                    num=checkpoint["num"], \
                    n_heads=checkpoint["n_heads"], \
                    max_encoder_len=checkpoint["max_encoder_len"], \
                    max_decoder_len=checkpoint["max_decoder_len"], \
                    use_conv=True)
        if cuda:
                model.cuda()
        
        if amp is not None:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
            amp.load_state_dict(checkpoint['amp'])
            #optimizer.load_state_dict(checkpoint['optimizer']) # dynamic loss scaling spikes if we load this! waiting for fix from nvidia apex
            print("Loaded amp state dict!")
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
            optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
        return model, optimizer
    
    def save_state(self, epoch, optimizer, scheduler, best_acc, path, amp=None):
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
                    'max_decoder_len': self.max_decoder_len,\
                    'amp': amp.state_dict()
                }
        torch.save(state, path)