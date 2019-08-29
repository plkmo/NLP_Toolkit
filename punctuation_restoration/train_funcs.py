# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:11:29 2019

@author: tsd
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from .models.Transformer import PuncTransformer
from .utils.misc import load_pickle, save_as_pickle, CosineWithRestarts
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_model_and_optimizer(args, src_vocab_size, trg_vocab_size, trg2_vocab_size, max_features_length,\
                             max_seq_length, mappings, idx_mappings, cuda):
    '''Loads the model (Transformer or encoder-decoder) based on provided arguments and parameters'''
    
    if args.model_no == 0:
        logger.info("Loading PuncTransformer...")        
        net = PuncTransformer(src_vocab=src_vocab_size, trg_vocab=trg_vocab_size, trg_vocab2=trg2_vocab_size, \
                              d_model=args.d_model, ff_dim=args.ff_dim,\
                                num=args.num, n_heads=args.n_heads, max_encoder_len=max_features_length, \
                                max_decoder_len=max_seq_length, mappings=mappings, idx_mappings=idx_mappings)
    '''
    elif args.model_no == 1:
        logger.info("Loading encoder-decoder (LAS) model...")
        net = LAS(vocab_size=vocab_size, listener_embed_size=args.LAS_embed_dim, listener_hidden_size=args.LAS_hidden_size, \
                  output_class_dim=vocab_size, max_label_len=max_seq_length)
    ''' 
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    criterion = nn.CrossEntropyLoss(ignore_index=1) # ignore padding tokens
    
    #model = SummaryTransformer if (args.model_no == 0) else LAS
    net, optimizer, scheduler, start_epoch, acc = load_state(net, args, load_best=False, load_scheduler=False)

    if cuda:
        net.cuda()

    return net, criterion, optimizer, scheduler, start_epoch, acc

def load_state(net, args, load_best=False, load_scheduler=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        if load_best:
            net = net.load_model(best_path)
        else:
            net = net.load_model(checkpoint_path)
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = CosineWithRestarts(optimizer, T_max=300)
        if load_scheduler:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = CosineWithRestarts(optimizer, T_max=300)
    return net, optimizer, scheduler, start_epoch, best_pred

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./data/test_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch
