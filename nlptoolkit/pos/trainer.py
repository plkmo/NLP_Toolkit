# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:58:17 2019

@author: WT
"""
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_model_and_optimizer, evaluate_results, load_results, decode_outputs
from .utils.misc_utils import load_pickle, save_as_pickle
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def train_and_fit(args):
    
    cuda = torch.cuda.is_available()
    
    train_loader, train_length, test_loader, test_length = load_dataloaders(args)
    
    vocab = load_pickle("vocab.pkl")
    logger.info("POS Vocabulary size: %d" % (len(vocab.pos2idx) - 1))
    
    ignore_idx = CrossEntropyLoss().ignore_index
    
    logger.info("Loading model and optimizers...")
    net, criterion, optimizer, scheduler, start_epoch, acc = load_model_and_optimizer(args, len(train_loader), cuda)
    
    '''
    ### freeze all layers except for last encoder layer and classifier layer
    logger.info("FREEZING MOST HIDDEN LAYERS...")
    unfrozen_layers = ["classifier", "bert.pooler", ]#"bert.encoder.layer.11", "bert.encoder.layer.10", "bert.encoder.layer.9",\
                       #"bert.encoder.layer.8", "bert.encoder.layer.7", "bert.encoder.layer.6", "bert.encoder.layer.5"]
    for name, param in net.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
            param.requires_grad = True
    '''
    
    losses_per_epoch, accuracy_per_epoch = load_results(model_no=args.model_no)
    
    batch_update_steps = int(len(train_loader)/10)
    logger.info("Starting training process...")
    net.zero_grad()
    for e in range(start_epoch, args.num_epochs):
        
        losses_per_batch = []; total_loss = 0.0
        for i, data in enumerate(train_loader):
            net.train()
            if args.model_no == 0:
                #labels = data[1].contiguous().view(-1)
                if len(data) == 4:
                    src_input = data[0]
                    src_mask = data[1]
                    token_type = data[2]
                    labels = data[3]
                else:
                    src_input = data[0]
                    labels = data[1]
                    src_mask = (src_input != 0).long()
                    token_type = torch.zeros((src_input.shape[0], src_input.shape[1]), dtype=torch.long)
        
                if cuda:
                    src_input = src_input.cuda().long(); labels = labels.cuda().long()
                    src_mask = src_mask.cuda(); token_type=token_type.cuda()
                outputs = net(src_input, attention_mask=src_mask, token_type_ids=token_type, labels=labels)
                #outputs = outputs[0]; #print(outputs[0,0,:])
                loss = outputs[0]
                
            elif args.model_no == 1:
                src_input, trg_input = data[0], data[1][:, :-1]
                labels = data[1][:,1:].contiguous().view(-1)
                if cuda:
                    src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                outputs = net(src_input, trg_input)
            
            #print(outputs.shape); print(labels.shape)
            #outputs = outputs.view(-1, outputs.size(-1))
            #loss = criterion(outputs, labels);
            loss = loss/args.gradient_acc_steps
            loss.backward();
            clip_grad_norm_(net.parameters(), args.max_norm)
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                scheduler.step()
                net.zero_grad()
            total_loss += loss.item()
            if i % batch_update_steps == (batch_update_steps - 1): # print every (batch_update_steps) mini-batches of size = batch_size
                losses_per_batch.append(args.gradient_acc_steps*total_loss/batch_update_steps)
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.7f' %
                      (e, (i + 1)*args.batch_size, train_length, losses_per_batch[-1]))
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(evaluate_results(net, test_loader if test_loader is not None else train_loader, \
                                                   cuda, None, None, args, ignore_idx,\
                                                   vocab.idx2pos)['accuracy'])
        print("Losses at Epoch %d: %.7f" % (e, losses_per_epoch[-1]))
        print("Accuracy at Epoch %d: %.7f" % (e, accuracy_per_epoch[-1]))
        
        decode_outputs(outputs[1].view(-1, outputs[1].size(-1)), labels.contiguous().view(-1), vocab.idx2pos, args)
                
        if accuracy_per_epoch[-1] > acc:
            acc = accuracy_per_epoch[-1]
            torch.save({
                    'epoch': e,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/" , "test_model_best_%d.pth.tar" % args.model_no))

        if (e % 1) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no, accuracy_per_epoch)
            torch.save({
                    'epoch': e,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/" , "test_checkpoint_%d.pth.tar" % args.model_no))

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