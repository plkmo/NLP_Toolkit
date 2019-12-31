# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 16:58:24 2019

@author: WT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class pBLSTMLayer(nn.Module):
    def __init__(self,input_size, lstm_hidden_size, num_layers=1):
        super(pBLSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_size=2*input_size, hidden_size=lstm_hidden_size,\
                             num_layers=num_layers, dropout=0.1, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        # X = batch X seq_len X features
        x = x.contiguous().view(x.shape[0], int(x.shape[1]/2), 2*x.shape[2])
        out, hidden = self.lstm(x)
        return out, hidden

class Listener(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Listener, self).__init__()
        self.layer1 = pBLSTMLayer(input_size, hidden_size)
        self.layer2 = pBLSTMLayer(2*hidden_size, hidden_size)
        self.layer3 = pBLSTMLayer(2*hidden_size, hidden_size)
    
    def forward(self, x):
        output, _ = self.layer1(x); #print("Listener output1:", output.shape)
        output, _ = self.layer2(output); #print("Listener output2:", output.shape)
        output, _ = self.layer3(output); #print("Listener output3:", output.shape)
        return output
    
    def flatten_parameters(self):
        self.layer1.lstm.flatten_parameters()
        self.layer2.lstm.flatten_parameters()
        self.layer3.lstm.flatten_parameters()

def Attention(queries, values):
    batch_size = queries.size(0)
    input_lengths = values.size(1) #125
    # (N, To, H) * (N, H, Ti) -> (N, To, Ti)
    attention_scores = torch.bmm(queries, values.transpose(1, 2)); #print("attention_scores", attention_scores.shape)
    attention_distribution = F.softmax(
        attention_scores.view(-1, input_lengths), dim=1).view(batch_size, -1, input_lengths); #print("attn_dist", attention_distribution.shape)
    # (batch_size X num_layers X downsampled seq_len) X (batch_size X downsampled seq_len X hidden_size) = (batch X num_layers X hidden)
    context = torch.bmm(attention_distribution, values)
    return context, attention_distribution

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size, features_size, d_model, droprate=0.1):
        super(AttentionBlock, self).__init__()
        self.d_model = d_model
        # learning layers for q,k,v
        self.s_matrix = nn.Linear(hidden_size, d_model)
        self.h_matrix = nn.Linear(features_size, d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        
    def forward(self, s_i, h):
        # s_i = batch_size X last layer (1) X hidden_size
        # h = batch_size X 8 times downsampled seq_len X 2*hidden_size
        s_i = self.s_matrix(s_i); #print("s_i", s_i.shape) # batch_size X d_model
        h = self.h_matrix(h); #print("h", h.shape) # batch_size X seq_len X d_model
        context, attention_distribution = Attention(s_i, h)
        
        context = self.fc1(context)
        return context, attention_distribution   

class Speller(nn.Module):
    def __init__(self, embed_dim, hidden_size, features_size, max_label_len, output_class_dim,\
                 d_model=64, num_layers=2):
        super(Speller, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(output_class_dim, embed_dim)
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(input_size=embed_dim + d_model, hidden_size=hidden_size)]
        self.rnn += [nn.LSTMCell(hidden_size, hidden_size) for _ in range(1, num_layers)]
        self.attention = AttentionBlock(hidden_size, features_size, d_model)
        self.max_label_len = max_label_len
        self.label_dim = output_class_dim
        self.character_distribution = nn.Linear(hidden_size + d_model, output_class_dim)
        self.teacher_forcing = True
            
    def forward(self, listener_feature, trg_input, infer=False):
        '''If infer=False, performs a forward pass and returns logits, else if 
        infer=True, performs forward passes with initial trg_input=<sos>, whose output is appended to generate the next sequence until
        <eos> or max_length is reached, returns the logits sequence'''
        pred_y = []
        if not self.teacher_forcing:
            y = [self.embed(torch.FloatTensor(np.zeros((listener_feature.shape[0], 1))))] # initialize sos token
        else:
            y = self.embed(trg_input); #print("y", y.shape); print("y1", y[:,1,:].shape)
            
        hidden_states = [torch.zeros((listener_feature.shape[0], self.hidden_size)) for _ in range(self.num_layers)]
        c_states = [torch.zeros((listener_feature.shape[0], self.hidden_size)) for _ in range(self.num_layers)]
        if listener_feature.is_cuda:
            hidden_states = [hidden_state.cuda() for hidden_state in hidden_states]
            c_states = [c_state.cuda() for c_state in c_states]
        
        if not infer:
            for step in range(len(y[0,:,0])):
                context, attention_score = self.attention(hidden_states[-1].unsqueeze(1), listener_feature); #print("Context", context.shape)
                rnn_input = torch.cat([y[:,step,:], context.squeeze(1)], dim=1); #print("rnn_input", rnn_input.shape)
                
                for layer in range(self.num_layers):
                    if layer == 0:
                        hidden_states[0], c_states[0] = self.rnn[layer](rnn_input, (hidden_states[0], c_states[0])); #print("hidden_state", hidden_states[0].shape)
                    else:
                        hidden_states[layer], c_states[layer] = self.rnn[layer](hidden_states[layer-1], (hidden_states[layer], c_states[layer]))
                
                logits = self.character_distribution(torch.cat([hidden_states[-1], context.squeeze(1)], dim=1)); #print("logits", logits.shape)
                if not self.teacher_forcing:
                    y.append(self.embed(torch.softmax(logits, dim=1).max(1)[1]))
                pred_y.append(logits)
            
            pred_y = torch.stack(pred_y, dim=1); #print("pred_y", pred_y.shape)
        
        else:
            for step in range(self.max_label_len):
                context, attention_score = self.attention(hidden_states[-1].unsqueeze(1), listener_feature); #print("Context", context.shape)
                rnn_input = torch.cat([y[:,step,:], context.squeeze(1)], dim=1); #print("rnn_input", rnn_input.shape)
                
                for layer in range(self.num_layers):
                    if layer == 0:
                        hidden_states[0], c_states[0] = self.rnn[layer](rnn_input, (hidden_states[0], c_states[0])); #print("hidden_state", hidden_states[0].shape)
                    else:
                        hidden_states[layer], c_states[layer] = self.rnn[layer](hidden_states[layer-1], (hidden_states[layer], c_states[layer]))
                
                logits = self.character_distribution(torch.cat([hidden_states[-1], context.squeeze(1)], dim=1)); #print("logits", logits.shape)
                pred_y.append(logits)
                pred_token = torch.softmax(logits, dim=1).max(1)[1]; #print(pred_token)
                y = torch.cat([y, self.embed(pred_token).unsqueeze(1)], dim=1)
                if pred_token.item() == 2:
                    break
            pred_y = torch.stack(pred_y, dim=1); #print("pred_y", pred_y.shape)
            pred_y = torch.softmax(pred_y, dim=2).max(2)[1]
        
        return pred_y
    
    def flatten_parameters(self):
        for l in range(len(self.rnn)):
            self.rnn[l].flatten_parameters()

class LAS(nn.Module):
    def __init__(self, vocab_size, listener_embed_size, listener_hidden_size, output_class_dim, \
                 max_label_len = 100):
        super(LAS, self).__init__()
        self.vocab_size = vocab_size
        self.listener_embed_size = listener_embed_size
        self.listener_hidden_size = listener_hidden_size
        self.output_class_dim = output_class_dim
        self.max_label_len = max_label_len
        self.embed = nn.Embedding(vocab_size, listener_embed_size)
        self.listener = Listener(listener_embed_size, listener_hidden_size)
        self.speller = Speller(listener_hidden_size, listener_hidden_size, 2*listener_hidden_size, max_label_len, \
                               output_class_dim, num_layers=2)
    
    def forward(self, x, trg_input, infer=False):
        # x = batch_size X seq_len
        x = self.embed(x)
        x = self.listener(x)
        x = self.speller(x, trg_input, infer)
        return x
    
    @classmethod
    def load_model(cls, path, args, cuda=True, amp=None):
        checkpoint = torch.load(path)
        model = cls(vocab_size=checkpoint['vocab_size'],\
                    listener_embed_size=checkpoint['listener_embed_size'], \
                    listener_hidden_size=checkpoint['listener_hidden_size'], \
                    output_class_dim=checkpoint['output_class_dim'],\
                    max_label_len=checkpoint['max_label_len'])
        model.listener.flatten_parameters()
        #model.speller.flatten_parameters()
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
                    'vocab_size': self.vocab_size,\
                    'listener_embed_size' : self.listener_embed_size,\
                    'listener_hidden_size': self.listener_hidden_size,\
                    'output_class_dim': self.output_class_dim,\
                    'max_label_len': self.max_label_len,\
                    'amp': amp.state_dict() if amp != None else None
                }
        torch.save(state, path)
        