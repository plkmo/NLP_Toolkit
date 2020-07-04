# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:58:01 2019

@author: WT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GIN(nn.Module):
    def __init__(self, X_size, n_nodes, A_hat, cuda, args, bias=True): # X_size = num features
        super(GIN, self).__init__()
        self.X_size = X_size
        self.n_nodes = n_nodes
        self.dum = torch.ones(size=(1, self.n_nodes))
        
        self.I = torch.eye(self.n_nodes, self.n_nodes)
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float()
        if cuda:
            self.A_hat = self.A_hat.cuda()
            self.I = self.I.cuda()
            self.dum = self.dum.cuda()
            
        self.diag_A = torch.diag(self.A_hat.diag())
        self.off_diag_A = self.A_hat - self.diag_A
        
        self.e1 = nn.parameter.Parameter(torch.zeros(size=(1, 1))) # size=(self.n_nodes, 1)
        var = 2./(self.e1.size(1) + self.e1.size(0))
        self.e1.data.normal_(0, var)
        self.mlp1_fc1 = nn.Linear(self.X_size, self.X_size//2, bias=bias)
        self.mlp1_fc2 = nn.Linear(self.X_size//2, self.X_size//4, bias=bias)
        
        '''
        self.e2 = nn.parameter.Parameter(torch.zeros(size=(1, 1))) # size=(self.n_nodes, 1)
        var2 = 2./(self.e2.size(1) + self.e2.size(0))
        self.e2.data.normal_(0, var2)
        self.mlp2_fc1 = nn.Linear(self.X_size, self.X_size//2, bias=bias)
        self.mlp2_fc2 = nn.Linear(self.X_size//2, self.X_size, bias=bias)
        '''
        self.fc1 = nn.Linear(self.X_size//4, args.num_classes, bias=bias)
        
    def forward(self, X): ### 1-layer GIN architecture
        hv = torch.mm(self.diag_A, X)
        hu = torch.mm(self.off_diag_A, X)
        
        X = torch.mm((torch.diag((self.dum*self.e1).squeeze()) + self.I), hv) + hu
        X = F.relu(self.mlp1_fc2(F.relu(self.mlp1_fc1(X))))
        
        '''
        hv = torch.mm(self.diag_A, X)
        hu = torch.mm(self.off_diag_A, X)
        X = torch.mm((torch.diag((self.dum*self.e2).squeeze()) + self.I), hv) + hu
        X = self.mlp2_fc2(F.relu(self.mlp2_fc1(X)))
        '''
        return self.fc1(X)
    
class GIN_batched(nn.Module):
    # TODO - still have to batch according to nearest neighbour!!!!
    def __init__(self, X_size, args, bias=True): # X_size = num features
        super(GIN_batched, self).__init__()
        self.X_size = X_size
        
        self.e1 = nn.parameter.Parameter(torch.zeros(size=(1, 1))) # size=(self.n_nodes, 1)
        var = 2./(self.e1.size(1) + self.e1.size(0))
        self.e1.data.normal_(0, var)
        self.mlp1_fc1 = nn.Linear(self.X_size, self.X_size, bias=bias)
        self.mlp1_fc2 = nn.Linear(self.X_size, self.X_size, bias=bias)
        
        '''
        self.e2 = nn.parameter.Parameter(torch.zeros(size=(1, 1))) # size=(self.n_nodes, 1)
        var2 = 2./(self.e2.size(1) + self.e2.size(0))
        self.e2.data.normal_(0, var2)
        self.mlp2_fc1 = nn.Linear(self.X_size, self.X_size, bias=bias)
        self.mlp2_fc2 = nn.Linear(self.X_size, self.X_size, bias=bias)
        '''
        self.fc1 = nn.Linear(self.X_size, args.num_classes, bias=bias)
        
    def forward(self, X, A_hat): ### 1-layer GIN architecture
        n_nodes = X.shape[0]
        dum = torch.ones(size=(1, n_nodes))
        
        I = torch.eye(n_nodes, n_nodes)
        #A_hat = torch.tensor(A_hat, requires_grad=False).float()
        if X.is_cuda:
            #A_hat = A_hat.cuda()
            I = I.cuda()
            dum = dum.cuda()
            
        diag_A = torch.diag(A_hat.diag())
        off_diag_A = A_hat - diag_A
        
        hv = torch.mm(diag_A, X)
        hu = torch.mm(off_diag_A, X)
        
        X = torch.mm((torch.diag((dum*self.e1).squeeze()) + I), hv) + hu
        X = F.relu(self.mlp1_fc2(F.relu(self.mlp1_fc1(X))))
        '''
        hv = torch.mm(diag_A, X)
        hu = torch.mm(off_diag_A, X)
        X = torch.mm((torch.diag((dum*self.e2).squeeze()) + I), hv) + hu
        X = self.mlp2_fc2(F.relu(self.mlp2_fc1(X)))
        '''
        return self.fc1(X)