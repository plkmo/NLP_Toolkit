# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:58:01 2019

@author: WT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, X_size, args, bias=True): # X_size = num features
        super(GCN, self).__init__()
        self.weight = nn.parameter.Parameter(torch.zeros(size=(X_size, args.hidden_size_1)))
        var = 2./(self.weight.size(1) + self.weight.size(0))
        self.weight.data.normal_(0, var)
        
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(args.hidden_size_1))
            self.bias.data.normal_(0, var)
        else:
            self.register_parameter("bias", None)
        
    def forward(self, X, A_hat): ### 1-layer GCN architecture
        X = torch.mm(X, self.weight)
        if self.bias is not None:
            X = (X + self.bias)
        X = F.relu(torch.mm(A_hat, X))
        return X
    
class DGI(nn.Module):
    def __init__(self, X_size, args, bias=True):
        super(DGI, self).__init__()
        self.encoder = GCN(X_size, args, bias=bias)
        self.D_weight = nn.parameter.Parameter(torch.zeros(size=(args.hidden_size_1,\
                                                                 args.hidden_size_1))) # features X features
         
    def summarize_patch(self, X):
        X = torch.sigmoid(X.mean(dim=0))
        return X
    
    def forward(self, X, A_hat, X_c):
        X = self.encoder(X, A_hat) # nodes X features
        X_c = self.encoder(X_c, A_hat) # nodes X features
        s = self.summarize_patch(X) # s = features
        fac = torch.mm(self.D_weight, s.unsqueeze(-1)) # fac = features X 1
        
        pos_D = []
        for i in range(X.shape[0]):
            pos_d_i = torch.sigmoid(torch.mm(X[i, :].unsqueeze(0), fac))
            pos_D.append(pos_d_i)
        pos_D = torch.stack(pos_D, dim=0).squeeze()
        
        neg_D = []
        for i in range(X_c.shape[0]):
            neg_d_i = torch.sigmoid(torch.mm(X_c[i, :].unsqueeze(0), fac))
            neg_D.append(neg_d_i)
        neg_D = torch.stack(neg_D, dim=0).squeeze()
        return pos_D, neg_D