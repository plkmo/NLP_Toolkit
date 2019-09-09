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

def create_trg_mask(trg, ignore_idx=1):
    trg_mask = (trg != ignore_idx).unsqueeze(-2)
    np_mask = np.triu(np.ones((1, trg.size(1),trg.size(1))),k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if trg.is_cuda:
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
    def __init__(self, d_model, n_heads, ff_dim, droprate=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.attn = MHAttention(d_model=d_model, n_heads=n_heads)
        self.dropout1 = nn.Dropout(droprate)
        self.norm2 = LayerNorm(d_model)
        self.fc1 = FeedForward(d_model=d_model, hidden_size=ff_dim)    
        self.dropout2 = nn.Dropout(droprate)
    
    def forward(self, x, mask):
        x1 = self.norm1(x); #print("e1", x1.shape)
        x = x + self.dropout1(self.attn(x1, x1, x1, mask)); #print("e2", x.shape)
        x1 = self.norm2(x)
        x = x + self.dropout2(self.fc1(x1)); #print("e3", x.shape)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, droprate=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(droprate)
        self.dropout2 = nn.Dropout(droprate)
        self.dropout3 = nn.Dropout(droprate)
        self.attn1 = MHAttention(d_model=d_model, n_heads=n_heads)
        self.attn2 = MHAttention(d_model=d_model, n_heads=n_heads)
        self.fc1 = FeedForward(d_model=d_model, hidden_size=ff_dim)
        
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
    def __init__(self, vocab_size, d_model, ff_dim, num, n_heads, max_encoder_len):
        super(EncoderBlock, self).__init__()
        self.num = num
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = Pos_Encoder(d_model, max_len=max_encoder_len)
        self.layers = clone_layers(EncoderLayer(d_model, n_heads, ff_dim), num)
        self.norm = LayerNorm(d_model)
    
    def forward(self, src, mask):
        x = self.embed(src); #print("e_embed", x.shape)
        x = self.pe(x)
        for i in range(self.num):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, vocab_size, d_model, ff_dim, num, n_heads, max_decoder_len):
        super(DecoderBlock, self).__init__()
        self.num = num
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = Pos_Encoder(d_model, max_len=max_decoder_len)
        self.layers = clone_layers(DecoderLayer(d_model, n_heads, ff_dim), num)
        self.norm = LayerNorm(d_model)
    
    def forward(self, trg, e_out, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.num):
            x = self.layers[i](x, e_out, src_mask, trg_mask)
        x = self.norm(x)
        return x

class PuncTransformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, trg_vocab2, d_model, ff_dim, num, n_heads,\
                 max_encoder_len, max_decoder_len, mappings, idx_mappings):
        super(PuncTransformer, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.trg_vocab2 = trg_vocab2
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num = num
        self.n_heads = n_heads
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.mappings = mappings
        self.idx_mappings = idx_mappings
        self.encoder = EncoderBlock(vocab_size=src_vocab, d_model=d_model, ff_dim=ff_dim,\
                                    num=num, n_heads=n_heads, max_encoder_len=max_encoder_len)
        self.decoder = DecoderBlock(vocab_size=trg_vocab, d_model=d_model, ff_dim=ff_dim,\
                                    num=num, n_heads=n_heads, max_decoder_len=max_decoder_len)
        self.decoder2 = DecoderBlock(vocab_size=trg_vocab2, d_model=d_model, ff_dim=ff_dim,\
                                    num=num, n_heads=n_heads, max_decoder_len=max_decoder_len)
        self.fc1 = nn.Linear(d_model, trg_vocab)
        self.fc2 = nn.Linear(d_model, trg_vocab2)
    
    def forward(self, src, trg, trg2, src_mask, trg_mask=None, trg_mask2=None, infer=False, trg_vocab_obj=None, \
                trg2_vocab_obj=None):
        e_out = self.encoder(src, src_mask); #print("e_out", e_out.shape)
        
        if infer == False:
            d_out = self.decoder(trg, e_out, src_mask, trg_mask); #print("d_out", d_out.shape)
            x = self.fc1(d_out); #print("x", x.shape)
            
            d_out2 = self.decoder2(trg2, e_out, src_mask, trg_mask2)
            x2 = self.fc2(d_out2)
            return x, x2
        else:
            stepwise_translated_words = []; stepwise_translated_word_idxs = []
            stepwise_translated_words2 = []; stepwise_translated_word_idxs2 = []
            cuda = src.is_cuda
            for i in range(2, self.max_decoder_len):
                trg_mask = create_trg_mask(trg, ignore_idx=1)
                trg2_mask = create_trg_mask(trg2, ignore_idx=self.idx_mappings['pad'])
                if cuda:
                    trg = trg.cuda(); trg_mask = trg_mask.cuda(); trg2 = trg2.cuda(); trg2_mask = trg2_mask.cuda()
                outputs = self.fc1(self.decoder(trg, e_out, src_mask, trg_mask))
                outputs2 = self.fc2(self.decoder2(trg2, e_out, src_mask, trg2_mask))
                out_idxs = torch.softmax(outputs, dim=2).max(2)[1]
                out_idxs2 = torch.softmax(outputs2, dim=2).max(2)[1]
                trg = torch.cat((trg, out_idxs[:,-1:]), dim=1)
                trg2 = torch.cat((trg2, out_idxs2[:,-1:]), dim=1)
                if cuda:
                    out_idxs = out_idxs.cpu().numpy(); out_idxs2 = out_idxs2.cpu().numpy()
                else:
                    out_idxs = out_idxs.numpy(); out_idxs2 = out_idxs2.numpy()
                stepwise_translated_word_idxs.append(out_idxs.tolist()[0][-1])
                stepwise_translated_word_idxs2.append(out_idxs2.tolist()[0][-1])
                if stepwise_translated_word_idxs[-1] == trg_vocab_obj.word_vocab['__eos']: # trg_vocab_obj = FR
                    break
                if stepwise_translated_word_idxs2[-1] == trg2_vocab_obj.punc2idx['eos']: # <eos> for label2
                    break
                stepwise_translated_words.append(next(trg_vocab_obj.inverse_transform([[stepwise_translated_word_idxs[-1]]])))
                stepwise_translated_words2.append(trg2_vocab_obj.idx2punc[stepwise_translated_word_idxs2[-1]])
            final_step_words = next(trg_vocab_obj.inverse_transform([out_idxs[0][:-1].tolist()]))
            final_step_words2 = [trg2_vocab_obj.idx2punc[i] for i in out_idxs2[0][:-1]]
            return stepwise_translated_words, final_step_words, stepwise_translated_words2, final_step_words2
    
    @classmethod # src_vocab, trg_vocab, d_model, num, n_heads
    def load_model(cls, path):
        checkpoint = torch.load(path)
        model = cls(src_vocab=checkpoint["src_vocab"], \
                    trg_vocab=checkpoint["trg_vocab"], \
                    trg_vocab2=checkpoint["trg_vocab2"], \
                    d_model=checkpoint["d_model"], \
                    ff_dim=checkpoint["ff_dim"], \
                    num=checkpoint["num"], \
                    n_heads=checkpoint["n_heads"], \
                    max_encoder_len=checkpoint["max_encoder_len"], \
                    max_decoder_len=checkpoint["max_decoder_len"], \
                    mappings=checkpoint["mappings"],\
                    idx_mappings=checkpoint["idx_mappings"])
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
                    'trg_vocab2': self.trg_vocab2, \
                    'd_model': self.d_model,\
                    'ff_dim': self.ff_dim,\
                    'num': self.num,\
                    'n_heads': self.n_heads,\
                    'max_encoder_len': self.max_encoder_len,\
                    'max_decoder_len': self.max_decoder_len,\
                    'mappings': self.mappings,\
                    'idx_mappings': self.idx_mappings
                }
        torch.save(state, path)


class PuncTransformer2(nn.Module):
    def __init__(self, src_vocab, trg_vocab, trg_vocab2, d_model, ff_dim, num, n_heads,\
                 max_encoder_len, max_decoder_len, mappings, idx_mappings):
        super(PuncTransformer2, self).__init__()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.trg_vocab2 = trg_vocab2
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.num = num
        self.n_heads = n_heads
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.mappings = mappings
        self.idx_mappings = idx_mappings
        self.encoder = EncoderBlock(vocab_size=src_vocab, d_model=d_model, ff_dim=ff_dim,\
                                    num=num, n_heads=n_heads, max_encoder_len=max_encoder_len)
        self.decoder = DecoderBlock(vocab_size=trg_vocab, d_model=d_model, ff_dim=ff_dim,\
                                    num=num, n_heads=n_heads, max_decoder_len=max_decoder_len)
        self.fc1 = nn.Linear(d_model, trg_vocab)
        self.fc2 = nn.Linear(d_model, trg_vocab2)
    
    def forward(self, src, trg, trg2, src_mask, trg_mask=None, trg_mask2=None, infer=False, trg_vocab_obj=None, \
                trg2_vocab_obj=None):
        e_out = self.encoder(src, src_mask); #print("e_out", e_out.shape)
        
        if infer == False:
            d_out = self.decoder(trg, e_out, src_mask, trg_mask); #print("d_out", d_out.shape)
            x = self.fc1(d_out); #print("x", x.shape)
            x2 = self.fc2(d_out)
            return x, x2
        else:
            stepwise_translated_words = []; stepwise_translated_word_idxs = []
            stepwise_translated_words2 = []; stepwise_translated_word_idxs2 = []
            cuda = src.is_cuda
            for i in range(2, self.max_decoder_len):
                trg_mask = create_trg_mask(trg, ignore_idx=1)
                if cuda:
                    trg = trg.cuda(); trg_mask = trg_mask.cuda()
                d_out = self.decoder(trg, e_out, src_mask, trg_mask)
                outputs = self.fc1(d_out)
                outputs2 = self.fc2(d_out)
                out_idxs = torch.softmax(outputs, dim=2).max(2)[1]
                out_idxs2 = torch.softmax(outputs2, dim=2).max(2)[1]
                trg = torch.cat((trg, out_idxs[:,-1:]), dim=1)

                if cuda:
                    out_idxs = out_idxs.cpu().numpy(); out_idxs2 = out_idxs2.cpu().numpy()
                else:
                    out_idxs = out_idxs.numpy(); out_idxs2 = out_idxs2.numpy()
                stepwise_translated_word_idxs.append(out_idxs.tolist()[0][-1])
                stepwise_translated_word_idxs2.append(out_idxs2.tolist()[0][-1])
                if stepwise_translated_word_idxs[-1] == trg_vocab_obj.word_vocab['__eos']: # trg_vocab_obj = FR
                    break
                if stepwise_translated_word_idxs2[-1] == trg2_vocab_obj.punc2idx['eos']: # <eos> for label2
                    break
                stepwise_translated_words.append(next(trg_vocab_obj.inverse_transform([[stepwise_translated_word_idxs[-1]]])))
                stepwise_translated_words2.append(trg2_vocab_obj.idx2punc[stepwise_translated_word_idxs2[-1]])
            final_step_words = next(trg_vocab_obj.inverse_transform([out_idxs[0][:-1].tolist()]))
            final_step_words2 = [trg2_vocab_obj.idx2punc[i] for i in out_idxs2[0][:-1]]
            return stepwise_translated_words, final_step_words, stepwise_translated_words2, final_step_words2
    
    @classmethod # src_vocab, trg_vocab, d_model, num, n_heads
    def load_model(cls, path):
        checkpoint = torch.load(path)
        model = cls(src_vocab=checkpoint["src_vocab"], \
                    trg_vocab=checkpoint["trg_vocab"], \
                    trg_vocab2=checkpoint["trg_vocab2"], \
                    d_model=checkpoint["d_model"], \
                    ff_dim=checkpoint["ff_dim"], \
                    num=checkpoint["num"], \
                    n_heads=checkpoint["n_heads"], \
                    max_encoder_len=checkpoint["max_encoder_len"], \
                    max_decoder_len=checkpoint["max_decoder_len"], \
                    mappings=checkpoint["mappings"],\
                    idx_mappings=checkpoint["idx_mappings"])
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
                    'trg_vocab2': self.trg_vocab2, \
                    'd_model': self.d_model,\
                    'ff_dim': self.ff_dim,\
                    'num': self.num,\
                    'n_heads': self.n_heads,\
                    'max_encoder_len': self.max_encoder_len,\
                    'max_decoder_len': self.max_decoder_len,\
                    'mappings': self.mappings,\
                    'idx_mappings': self.idx_mappings
                }
        torch.save(state, path)