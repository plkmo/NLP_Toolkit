# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:58:48 2019

@author: WT
"""

import os
import torch
from torch.nn.utils import clip_grad_norm_
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_model_and_optimizer, evaluate_results, load_results, decode_outputs
from .utils import load_pickle, save_as_pickle, lrate
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def train_and_fit(args, pyTransformer=False):
    
    if pyTransformer:
        from .models.Transformer.py_Transformer import create_masks
    else:
        from .models.Transformer.transformer_model import create_masks
    
    logger.info("Loading data...")
    train_loader, train_length, max_features_length, max_seq_length = load_dataloaders(args)
    print("Max features length: %d" % max_features_length)
    print("Max sequence length: %d" % max_seq_length)
    vocab = load_pickle("vocab.pkl")
    
    if args.fp16:    
        from apex import amp
    else:
        amp = None
    
    logger.info("Loading model and optimizers...")
    cuda = torch.cuda.is_available()
    net, criterion, optimizer, scheduler, start_epoch, acc, g_mask1, g_mask2 = load_model_and_optimizer(args, vocab, \
                                                                                                        max_features_length, \
                                                                                                        max_seq_length, cuda,\
                                                                                                        pyTransformer)
    losses_per_epoch, accuracy_per_epoch = load_results(model_no=args.model_no)    
    batch_update_steps = int(train_length/(args.batch_size*10))
    
    optimizer.zero_grad()
    logger.info("Starting training process...")
    for e in range(start_epoch, args.num_epochs):
        #l_rate = lrate(e + 1, d_model=32, k=10, warmup_n=25000)
        net.train()
        losses_per_batch = []; total_loss = 0.0
        for i, data in enumerate(train_loader):
            
            if args.model_no == 0:
                src_input, trg_input, f_len = data[0], data[1][:, :-1], data[2]
                labels = data[1][:,1:].contiguous().view(-1)
                src_mask, trg_mask = create_masks(src_input, trg_input, f_len, args)
                if cuda:
                    src_input = src_input.cuda().float(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
                outputs = net(src_input, trg_input, src_mask, trg_mask, g_mask1, g_mask2)
                
            elif args.model_no == 1:
                src_input, trg_input = data[0], data[1][:, :-1]
                labels = data[1][:,1:].contiguous().view(-1)
                if cuda:
                    src_input = src_input.cuda().float(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                outputs = net(src_input, trg_input)
                    
            outputs = outputs.view(-1, outputs.size(-1))
            loss = criterion(outputs, labels);
            loss = loss/args.gradient_acc_steps
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
            else:
                grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
                
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            total_loss += loss.item()
            if i % batch_update_steps == (batch_update_steps - 1): # print every (batch_update_steps) mini-batches of size = batch_size
                losses_per_batch.append(args.gradient_acc_steps*total_loss/batch_update_steps)
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.7f' %
                      (e, (i + 1)*args.batch_size, train_length, losses_per_batch[-1]))
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(evaluate_results(net, train_loader, cuda, g_mask1, g_mask2, create_masks, args))
        print("Losses at Epoch %d: %.7f" % (e, losses_per_epoch[-1]))
        print("Accuracy at Epoch %d: %.7f" % (e, accuracy_per_epoch[-1]))
        decode_outputs(outputs, labels, vocab)
        if accuracy_per_epoch[-1] > acc:
            acc = accuracy_per_epoch[-1]
            net.save_state(epoch=(e+1), optimizer=optimizer, scheduler=scheduler, best_acc=acc,\
                           path=os.path.join("./data/" ,\
                    "test_model_best_%d.pth.tar" % args.model_no))

        if (e % 2) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no, accuracy_per_epoch)
            net.save_state(epoch=(e+1), optimizer=optimizer, scheduler=scheduler, best_acc=acc,\
                           path=os.path.join("./data/" ,\
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