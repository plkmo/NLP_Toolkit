# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:25:41 2019

@author: WT
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .train_funcs import load_datasets, load_state, load_results, evaluate, infer, batched_samples
from .GCN import gcn, gcn_batched
from .preprocessing_funcs import load_pickle, save_as_pickle
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_and_fit(args):
    cuda = torch.cuda.is_available()
    
    f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_datasets(args, train_test_split=args.train_test_split)
    targets = torch.tensor(labels_selected).long()
    #print(labels_selected, labels_not_selected)
    if args.batched == 0:
        net = gcn(X.shape[1], A_hat, cuda, args)
    elif args.batched == 1:
        doc_nodes = [a for a in range(len(labels_selected) + len(labels_not_selected))]
        net = gcn_batched(X.shape[1], args)
        train_set = batched_samples(X, A_hat, doc_nodes, args)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, \
                                  num_workers=0, pin_memory=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(args.num_epochs) \
                                                                      if ((i + 1) % 200) == 0], gamma=0.9)
    
    start_epoch, best_pred = load_state(net, optimizer, scheduler, model_no=args.model_no, load_best=False)
    losses_per_epoch, evaluation_trained, evaluation_untrained = load_results(model_no=args.model_no)
    
    if cuda:
        net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        f = f.cuda()
        targets = targets.cuda()
        
    logger.info("Starting training process...")
    net.train()
    for e in range(start_epoch, args.num_epochs):
        if args.batched == 0:
            optimizer.zero_grad()
            output = net(f)
            loss = criterion(output[selected], targets)
            losses_per_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        elif args.batched == 1:
            skips = 0
            losses_per_batch = []
            update_size = len(train_loader)//10

            for eidx, data in enumerate(train_loader):
                X_batched, A_batched, n_nodes = data
                X_batched, A_batched, n_nodes = X_batched.squeeze(), A_batched.squeeze(), \
                                                n_nodes.squeeze().tolist()
                if cuda:
                    X_batched = X_batched.cuda()
                    A_batched = A_batched.cuda()
                    
                selected_batched = list(set(selected).intersection(set(n_nodes)))
                selected_idx = [selected.index(sb) for sb in selected_batched]

                sel = []
                for idx, batch_idx in enumerate(n_nodes):
                    if batch_idx in selected_batched:
                        sel.append(idx)
                #return X_batched, A_batched, sel, selected_batched, n_nodes, selected, labels_selected, labels_not_selected
                if len(selected_batched) == 0:
                    skips += 1
                    continue
                
                optimizer.zero_grad()
                output = net(X_batched, A_batched)
                loss = criterion(output[sel], targets[selected_idx])
                losses_per_batch.append(loss.item())
                
                if (eidx % update_size) == 0:
                    logger.info("[Epoch %d]: Batch loss, batch_size (%d/%d): %.5f, %d" % (e,\
                                                                                    eidx,\
                                                                                    len(train_loader),\
                                                                                    losses_per_batch[-1],\
                                                                                    len(n_nodes)))
                
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            logger.info("Skipped batches due to no labels/too few samples in sampled data: %d" % skips)
            av_loss = sum(losses_per_batch)/len(losses_per_batch)
            logger.info('Averaged batch losses: %.3f' % av_loss)
            losses_per_epoch.append(av_loss)
        
        if e % 50 == 0:
            #print(output[selected]); print(targets)
            ### Evaluate other untrained nodes and check accuracy of labelling
            net.eval()
            with torch.no_grad():
                if args.batched == 0:
                    pred_labels = net(f)
                elif args.batched == 1:
                    pred_labels = net(f, torch.FloatTensor(A_hat).cuda() if cuda else torch.FloatTensor(A_hat))
                trained_accuracy = evaluate(pred_labels[selected], labels_selected); untrained_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
            evaluation_trained.append((e, trained_accuracy)); evaluation_untrained.append((e, untrained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (e, trained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f" % (e, untrained_accuracy))
            print("[Epoch %d]: Loss: %.7f" % (e, losses_per_epoch[-1]))
            
            if args.batched == 0:
                print("Labels of trained nodes: \n", output[selected].max(1)[1])
                print("Ground Truth Labels of trained nodes: \n", targets)
            elif args.batched == 1:
                print("Labels of trained nodes: \n", output[sel].max(1)[1])
                print("Ground Truth Labels of trained nodes: \n", targets[selected_idx])
                
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
        if (e % 250) == 0:
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
    
    infer(args, f, test_idxs, net, A_hat,)