# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:06:59 2019

@author: tsd
"""

import os
import torch
from torch.nn.utils import clip_grad_norm_
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_state, load_results, load_model_and_optimizer, evaluate_results, decode_outputs, decode_outputs_p
from .utils.word_char_level_vocab import tokener
from .utils.bpe_vocab import Encoder
from .utils.misc import save_as_pickle, load_pickle
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def train_and_fit(args):
    
    cuda = torch.cuda.is_available()
    #cuda = False
    
    df, train_loader, train_length, max_features_length, max_output_len = load_dataloaders(args)
    
    if args.model_no == 0:
        from .models.Transformer import create_masks, create_trg_mask
    elif args.model_no == 2:
        from .models.py_Transformer import create_masks, create_trg_mask
    else:
        create_masks, create_trg_mask = None, None
    
    if args.level == "bpe":
        vocab = Encoder.load("./data/vocab.pkl")
        vocab_size = len(vocab.bpe_vocab) + len(vocab.word_vocab)
        tokenizer_en = tokener("en")
        vocab.word_tokenizer = tokenizer_en.tokenize
        vocab.custom_tokenizer = True
        mappings = load_pickle("mappings.pkl") # {'!': 250, '?': 34, '.': 5, ',': 4}
        idx_mappings = load_pickle("idx_mappings.pkl") # {250: 0, 34: 1, 5: 2, 4: 3, 'word': 4, 'sos': 5, 'eos': 6, 'pad': 7}
        
        inv_idx = {v: k for k, v in idx_mappings.items()} # {0: 250, 1: 34, 2: 5, 3: 4, 4: 'word', 5: 'sos', 6: 'eos', 7: 'pad'}
        inv_map = {v:k for k, v in mappings.items()} # {250: '!', 34: '?', 5: '.', 4: ','}
        
    logger.info("Max features length = %d %ss" % (max_features_length, args.level))
    logger.info("Max output length = %d" % (max_output_len))
    logger.info("Vocabulary size: %d" % vocab_size)
    logger.info("Mappings length: %d" % len(mappings))
    logger.info("idx_mappings length: %d" % len(idx_mappings))
        
    logger.info("Loading model and optimizers...")
    net, criterion, optimizer, scheduler, start_epoch, acc = load_model_and_optimizer(args=args, src_vocab_size=vocab_size, \
                                                                                      trg_vocab_size=vocab_size,\
                                                                                      trg2_vocab_size=len(idx_mappings),\
                                                                                      max_features_length=args.max_encoder_len,\
                                                                                      max_seq_length=args.max_decoder_len, \
                                                                                      mappings=mappings,\
                                                                                      idx_mappings=idx_mappings,\
                                                                                      cuda=cuda)
    losses_per_epoch, accuracy_per_epoch = load_results(model_no=args.model_no)
    
    batch_update_steps = int(len(train_loader)/10)
    logger.info("Starting training process...")
    for e in range(start_epoch, args.num_epochs):
        #l_rate = lrate(e + 1, d_model=32, k=10, warmup_n=25000)
        net.train()
        losses_per_batch = []; total_loss = 0.0
        for i, data in enumerate(train_loader):
            
            if args.model_no == 0:
                src_input, trg_input, trg2_input = data[0], data[1][:, :-1], data[2][:, :-1]
                labels = data[1][:,1:].contiguous().view(-1)
                labels2 = data[2][:,1:].contiguous().view(-1)
                src_mask, trg_mask = create_masks(src_input, trg_input)
                trg2_mask = create_trg_mask(trg2_input, ignore_idx=idx_mappings['pad'])
                if cuda:
                    src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda(); trg2_mask = trg2_mask.cuda()
                    trg2_input = trg2_input.cuda().long(); labels2 = labels2.cuda().long()
                #return src_input, trg_input, trg2_input, src_mask, trg_mask, trg2_mask, labels, labels2
                outputs, outputs2 = net(src_input, trg_input, trg2_input, src_mask, trg_mask, trg2_mask)
                
            elif args.model_no == 1:
                src_input, trg_input, trg2_input = data[0], data[1][:, :-1], data[2][:, :-1]
                labels = data[1][:,1:].contiguous().view(-1)
                labels2 = data[2][:,1:].contiguous().view(-1)
                if cuda:
                    src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    trg2_input = trg2_input.cuda().long(); labels2 = labels2.cuda().long()
                outputs, outputs2 = net(src_input, trg_input, trg2_input)
                    
            outputs = outputs.view(-1, outputs.size(-1))
            outputs2 = outputs2.view(-1, outputs2.size(-1))
            #print(outputs.shape, outputs2.shape, labels.shape, labels2.shape)
            loss = criterion(outputs, outputs2, labels, labels2);
            loss = loss/args.gradient_acc_steps
            loss.backward();
            #clip_grad_norm_(net.parameters(), args.max_norm)
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
        accuracy_per_epoch.append(evaluate_results(net, train_loader, cuda, None, None, args, \
                                                   create_masks, create_trg_mask, ignore_idx2=idx_mappings['pad']))
        print("Losses at Epoch %d: %.7f" % (e, losses_per_epoch[-1]))
        print("Accuracy at Epoch %d: %.7f" % (e, accuracy_per_epoch[-1]))
        
        if (args.level == "word") or (args.level == "char"):
            decode_outputs(outputs, labels, vocab.convert_idx2w, args)
        elif args.level == "bpe":
            decode_outputs(outputs, labels, vocab.inverse_transform, args)
            decode_outputs_p(outputs2, labels2, inv_idx, inv_map)
        
        if accuracy_per_epoch[-1] > acc:
            acc = accuracy_per_epoch[-1]
            net.save_state(epoch=(e+1), optimizer=optimizer, scheduler=scheduler, best_acc=acc,\
                           path=os.path.join("./data/" ,\
                    "test_model_best_%d.pth.tar" % args.model_no))

        if (e % 1) == 0:
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