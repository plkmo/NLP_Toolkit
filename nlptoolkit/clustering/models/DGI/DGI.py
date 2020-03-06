# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:58:01 2019

@author: WT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, X_size, A_hat, cuda, args, bias=True): # X_size = num features
        super(GCN, self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float()
        if cuda:
            self.A_hat = self.A_hat.cuda()
        self.weight = nn.parameter.Parameter(torch.zeros(size=(X_size, args.hidden_size_1)))
        var = 2./(self.weight.size(1) + self.weight.size(0))
        self.weight.data.normal_(0, var)
        
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(args.hidden_size_1))
            self.bias.data.normal_(0, var)
        else:
            self.register_parameter("bias", None)
        
    def forward(self, X): ### 1-layer GCN architecture
        X = torch.mm(X, self.weight)
        if self.bias is not None:
            X = (X + self.bias)
        X = F.relu(torch.mm(self.A_hat, X))
        return X
    
class DGI(nn.Module):
    def __init__(self, X_size, A_hat, cuda, args, bias=True):
        super(DGI, self).__init__()
        self.encoder = GCN(X_size, A_hat, cuda, args, bias=bias)
        self.D_weight = nn.parameter.Parameter(torch.zeros(size=(X_size, X_size))) # nodes X features
         
    def summarize_patch(self, X):
        X = torch.sigmoid(X.mean(dim=1))
        return X