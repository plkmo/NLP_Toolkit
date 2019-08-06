# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:26:00 2019

@author: WT
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .models.Transformer import Transformer, create_masks
from .train_funcs import load_dataloaders, load_state, load_results, evaluate_results
from .utils import CosineWithRestarts, save_as_pickle
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def train_and_fit(args):

    train_iter, FR, EN, train_length = load_dataloaders(args)
    src_vocab = len(EN.vocab)
    trg_vocab = len(FR.vocab)
    
    cuda = torch.cuda.is_available()
    net = Transformer(src_vocab=src_vocab, trg_vocab=trg_vocab, d_model=args.d_model, num=args.num, n_heads=args.n_heads)
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=1)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,100,200], gamma=0.7)
    scheduler = CosineWithRestarts(optimizer, T_max=330)
    if cuda:
        net.cuda()
    start_epoch, acc = load_state(net, optimizer, scheduler, args.model_no, load_best=False)
    losses_per_epoch, accuracy_per_epoch = load_results(model_no=args.model_no)
    
    logger.info("Starting training process...")
    batch_update = 300
    
    for e in range(start_epoch, args.num_epochs):
        net.train()
        losses_per_batch = []; total_loss = 0.0
        for i, data in enumerate(train_iter):
            trg_input = data.FR[:,:-1]
            labels = data.FR[:,1:].contiguous().view(-1)
            src_mask, trg_mask = create_masks(data.EN, trg_input)
            if cuda:
                data.EN = data.EN.cuda(); trg_input = trg_input.cuda(); labels = labels.cuda()
                src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
            outputs = net(data.EN, trg_input, src_mask, trg_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            loss = criterion(outputs, labels)
            loss = loss/args.gradient_acc_steps
            loss.backward(); #break;break
            #clip_grad_norm_(net.parameters(), args.max_norm)
            if (e % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            total_loss += loss.item()
            if i % batch_update == (batch_update - 1): # print every 100 mini-batches of size = batch_size
                losses_per_batch.append(args.gradient_acc_steps*total_loss/batch_update)
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.7f' %
                      (e, (i + 1)*args.batch_size, train_length, losses_per_batch[-1]))
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(evaluate_results(net, train_iter, cuda))
        print("Losses at Epoch %d: %.7f" % (e, losses_per_epoch[-1]))
        print("Accuracy at Epoch %d: %.7f" % (e, accuracy_per_epoch[-1]))
        if accuracy_per_epoch[-1] > acc:
            acc = accuracy_per_epoch[-1]
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': acc,\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/" ,\
                    "test_model_best_%d.pth.tar" % args.model_no))
        if (e % 1) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no, accuracy_per_epoch)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/",\
                    "test_checkpoint_%d.pth.tar" % args.model_no))
    
    logger.info("Finished training")
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(losses_per_epoch))], losses_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_title("Loss vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/",\
                             "test_loss_vs_epoch_%d.png" % args.model_no))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(accuracy_per_epoch))], accuracy_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.set_title("Accuracy vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/",\
                             "test_Accuracy_vs_epoch_%d.png" % args.model_no))