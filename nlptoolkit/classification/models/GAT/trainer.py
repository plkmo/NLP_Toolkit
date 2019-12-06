# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:25:41 2019

@author: WT
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .train_funcs import load_datasets, load_state, load_results, evaluate, infer
from .GAT import GAT, SpGAT
from .preprocessing_funcs import load_pickle, save_as_pickle
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_and_fit(args, sparse=True):
    cuda = torch.cuda.is_available() and args.use_cuda
    
    f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_datasets(args, train_test_split=args.train_test_split)
    targets = torch.tensor(labels_selected).long()
    A_hat = torch.tensor(A_hat)
    
    if sparse:
        net = SpGAT(nfeat=X.shape[1], 
                    nhid=args.hidden, 
                    nclass=args.num_classes, 
                    dropout=0.1, 
                    nheads=args.nb_heads, 
                    alpha=0.2)
    else:
        net = GAT(nfeat=X.shape[1], 
                    nhid=args.hidden, 
                    nclass=args.num_classes, 
                    dropout=0.1, 
                    nheads=args.nb_heads, 
                    alpha=0.2)
    del X
    
    criterion = F.nll_loss #nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000,6000], gamma=0.77)
    
    start_epoch, best_pred = load_state(net, optimizer, scheduler, model_no=args.model_no, load_best=False)
    losses_per_epoch, evaluation_trained, evaluation_untrained = load_results(model_no=args.model_no)
    
    if cuda:
        net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        f = f.cuda(); A_hat = A_hat.cuda()
        targets = targets.cuda()
        
    logger.info("Starting training process...")
    net.train()
    save_step = 5
    for e in range(start_epoch, args.num_epochs):
        optimizer.zero_grad()
        output = net(f, A_hat)
        loss = criterion(output[selected], targets)
        losses_per_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if e % save_step == 0:
            ### Evaluate other untrained nodes and check accuracy of labelling
            net.eval()
            with torch.no_grad():
                pred_labels = net(f, A_hat)
                trained_accuracy = evaluate(pred_labels[selected], labels_selected); untrained_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
            evaluation_trained.append((e, trained_accuracy)); evaluation_untrained.append((e, untrained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (e, trained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f" % (e, untrained_accuracy))
            print("[Epoch %d]: Loss: %.7f" % (e, losses_per_epoch[-1]))
            print("Labels of trained nodes: \n", output[selected].max(1)[1])
            net.train()
            if trained_accuracy > best_pred:
                best_pred = trained_accuracy
                torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': trained_accuracy,\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/" ,\
                    "test_model_best_%d.pth.tar" % args.model_no))
            
        if (e % save_step) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no, evaluation_untrained)
            save_as_pickle("train_accuracy_per_epoch_%d.pkl" % args.model_no, evaluation_trained)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': trained_accuracy,\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/",\
                    "test_checkpoint_%d.pth.tar" % args.model_no))
        scheduler.step()
    
    logger.info("Finished training!")
    evaluation_trained = np.array(evaluation_trained); evaluation_untrained = np.array(evaluation_untrained)
    save_as_pickle("test_losses_per_epoch_%d_final.pkl" % args.model_no, losses_per_epoch)
    save_as_pickle("train_accuracy_per_epoch_%d_final.pkl" % args.model_no, evaluation_trained)
    save_as_pickle("test_accuracy_per_epoch_%d_final.pkl" % args.model_no, evaluation_untrained)
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(losses_per_epoch))], losses_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_title("Loss vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/", "loss_vs_epoch_%d.png" % args.model_no))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter(evaluation_trained[:,0], evaluation_trained[:,1])
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy on trained nodes", fontsize=15)
    ax.set_title("Accuracy (trained nodes) vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/", "trained_accuracy_vs_epoch_%d.png" % args.model_no))

    if len(labels_not_selected) > 0:    
        fig = plt.figure(figsize=(13,13))
        ax = fig.add_subplot(111)
        ax.scatter(evaluation_untrained[:,0], evaluation_untrained[:,1])
        ax.set_xlabel("Epoch", fontsize=15)
        ax.set_ylabel("Accuracy on untrained nodes", fontsize=15)
        ax.set_title("Accuracy (untrained nodes) vs Epoch", fontsize=20)
        plt.savefig(os.path.join("./data/", "untrained_accuracy_vs_epoch_%d.png" % args.model_no))
        
        fig = plt.figure(figsize=(13,13))
        ax = fig.add_subplot(111)
        ax.scatter(evaluation_trained[:,0], evaluation_trained[:,1], c="red", marker="v", \
                   label="Trained Nodes")
        ax.scatter(evaluation_untrained[:,0], evaluation_untrained[:,1], c="blue", marker="o",\
                   label="Untrained Nodes")
        ax.set_xlabel("Epoch", fontsize=15)
        ax.set_ylabel("Accuracy", fontsize=15)
        ax.set_title("Accuracy vs Epoch", fontsize=20)
        ax.legend(fontsize=20)
        plt.savefig(os.path.join("./data/", "combined_plot_accuracy_vs_epoch_%d.png" % args.model_no))
    
    infer(f, test_idxs, net, A_hat)
    
    return net