# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:25:41 2019

@author: WT
"""
import os
import numpy as np
import torch
import torch.optim as optim
from .train_funcs import load_datasets, get_X_A_hat, JSdiv_Loss,\
                        load_state, load_results, infer
from .DGI import DGI
from .preprocessing_funcs import load_pickle, save_as_pickle
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_and_fit(args):
    cuda = torch.cuda.is_available()
    
    G = load_datasets(args)
    X, A_hat = get_X_A_hat(G, corrupt=False)
    #print(labels_selected, labels_not_selected)
    net = DGI(X.shape[1], args)
    criterion = JSdiv_Loss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000,6000],\
                                               gamma=0.77)
    
    start_epoch, best_pred = load_state(net, optimizer, scheduler, model_no=args.model_no, load_best=False)
    losses_per_epoch = load_results(model_no=args.model_no)
    
    if cuda:
        net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        
    logger.info("Starting training process...")
    net.train()
    for e in range(start_epoch, args.num_epochs):
        
        X, A_hat = get_X_A_hat(G, corrupt=False)
        X_c, _ = get_X_A_hat(G, corrupt=True)
        
        if cuda:
            X, A_hat, X_c = X.cuda(), A_hat.cuda(), X_c.cuda()
            
        pos_D, neg_D = net(X, A_hat, X_c)
        loss = criterion(pos_D, neg_D)
        losses_per_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (e % 10) == 0:
            print('[Epoch: %d] total loss: %.3f' %
                      (e + 1, losses_per_epoch[-1]))
            save_as_pickle("train_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': losses_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/",\
                    "test_checkpoint_%d.pth.tar" % args.model_no))
        scheduler.step()
    
    logger.info("Finished training!")
    save_as_pickle("train_losses_per_epoch_%d_final.pkl" % args.model_no, losses_per_epoch)
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(losses_per_epoch))], losses_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_title("Loss vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/", "loss_vs_epoch_%d.png" % args.model_no))
    return net
    