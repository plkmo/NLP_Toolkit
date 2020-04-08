# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:25:41 2019

@author: WT
"""
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .train_funcs import load_datasets, X_A_hat, JSdiv_Loss,\
                        load_state, load_results, infer, batched_samples
from .DGI import DGI
from .preprocessing_funcs import load_pickle, save_as_pickle
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_and_fit(args):
    cuda = torch.cuda.is_available()
    
    G, doc_nodes = load_datasets(args)
    X_A = X_A_hat(G)
    X, A_hat = X_A.get_X_A_hat(corrupt=False)
    logger.info("Adj matrix stats (min, max, mean): %.5f, %.5f, %.5f" % (A_hat.min(),\
                                                                           A_hat.max(),\
                                                                           A_hat.mean()))
    logger.info("Number of nodes (doc nodes, total): %d, %d" % (len(doc_nodes), X.shape[0]))
    net = DGI(X.shape[1], args, bias=True,\
              n_nodes=X.shape[0], A_hat=A_hat, cuda=cuda)
    
    if args.batched == 1:
        train_loader = batched_samples(X, A_hat, doc_nodes, args)
        #train_loader = DataLoader(train_set, batch_size=1, shuffle=True, \
        #                          num_workers=0, pin_memory=False)
    
    criterion = JSdiv_Loss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000,5500],\
                                               gamma=0.77)
    
    start_epoch, best_pred = load_state(net, optimizer, scheduler, model_no=args.model_no, load_best=False)
    losses_per_epoch = load_results(model_no=args.model_no)
    
    if cuda:
        net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        
    logger.info("Starting training process...")
    net.train()
    for e in range(start_epoch, args.num_epochs):
        
        if args.batched == 0:
            X, A_hat = X_A.get_X_A_hat(corrupt=False)
            X_c, _ = X_A.get_X_A_hat(corrupt=True)
            
            if cuda:
                X, A_hat, X_c = X.cuda(), A_hat.cuda(), X_c.cuda()
                
            pos_D, neg_D = net(X, A_hat, X_c)
            loss = criterion(pos_D, neg_D)
            losses_per_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        elif args.batched == 1:
            skips = 0
            losses_per_batch = []
            update_size = len(train_loader)//10
            batch_counter = 0
            while True:
                X, A_hat, n_nodes = train_loader.__getitem__(0, corrupt=False)
                X_c, _, _ = train_loader.__getitem__(0, corrupt=True)
                
                if X.shape[0] != X_c.shape[0]:
                    skips += 1
                    continue
                
                if cuda:
                    X, A_hat, X_c = X.cuda(), A_hat.cuda(), X_c.cuda()
                
                pos_D, neg_D = net(X, A_hat, X_c)
                loss = criterion(pos_D, neg_D)
                losses_per_batch.append(loss.item())
                
                if (batch_counter % update_size) == 0:
                    logger.info("[Epoch %d]: Batch loss, batch_size (%d/%d): %.5f, %d" % (e + 1,\
                                                                                        batch_counter,\
                                                                                        len(train_loader),\
                                                                                        losses_per_batch[-1],\
                                                                                        len(n_nodes)))
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                if batch_counter == len(train_loader):
                    break
                batch_counter += 1
                
            logger.info("Skipped batches due to no labels/too few samples in sampled data: %d" % skips)
            av_loss = sum(losses_per_batch)/len(losses_per_batch)
            logger.info('Averaged batch losses: %.5f' % av_loss)
            losses_per_epoch.append(av_loss)
        
        if (e % 10) == 0:
            print('[Epoch: %d] total loss: %.5f' %
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
    