# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:41:08 2019

@author: WT
"""
import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import load_pickle, CosineWithRestarts

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def dum_tokenizer(sent):
    return sent.split()

def load_model_and_optimizer(args, src_vocab, trg_vocab, cuda, amp=None, pytransformer=False):
    '''Loads the model based on provided arguments and parameters'''
    logger.info("Initializing model...")
    
    if pytransformer:
        from .models.Transformer.py_Transformer import pyTransformer as Transformer
    else:
        from .models.Transformer.Transformer import Transformer
        
    net = Transformer(src_vocab=src_vocab, trg_vocab=trg_vocab, d_model=args.d_model, ff_dim=args.ff_dim,\
                      num=args.num, n_heads=args.n_heads, max_encoder_len=args.max_encoder_len,\
                      max_decoder_len=args.max_decoder_len)
        
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=1)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,100,200], gamma=0.7)
    net, optimizer, start_epoch, acc, loaded_opt = load_state(net, cuda, args, load_best=False, \
                                                              amp=amp)
    
    if cuda and (not loaded_opt):
        net.cuda()
        
    if (not loaded_opt):
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CosineWithRestarts(optimizer, T_max=args.T_max)
    
    if (args.fp16) and (not loaded_opt) and (amp is not None):
        logger.info("Using fp16...")
        net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
        scheduler = CosineWithRestarts(optimizer, T_max=args.T_max)

    logger.info("Done setting up model, optimizer and scheduler.")
   
    return net, criterion, optimizer, scheduler, start_epoch, acc

def load_state(net, cuda, args, load_best=False, amp=None):
    """ Loads saved model and optimizer states if exists """
    loaded_opt = False
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
            net, optimizer = net.load_model(best_path, args, cuda, amp)
        else:
            net, optimizer = net.load_model(checkpoint_path, args, cuda, amp)

        logger.info("Loaded model and optimizer.")
        loaded_opt = True
    else:
        optimizer = None
    return net, optimizer, start_epoch, best_pred, loaded_opt

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

def evaluate(output, labels):
    ### ignore index 1 (padding) when calculating accuracy
    idxs = (labels != 1).nonzero().squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]
    if len(idxs) > 1:
        return (labels[idxs] == o_labels[idxs]).sum().item()/len(idxs)
    else:
        return (labels[idxs] == o_labels[idxs]).sum().item()

def evaluate_results(net, data_loader, create_masks, cuda):
    acc = 0
    print("Evaluating...")
    with torch.no_grad():
        net.eval()
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            trg_input = data.FR[:,:-1]
            labels = data.FR[:,1:].contiguous().view(-1)
            src_mask, trg_mask = create_masks(data.EN, trg_input)
            if cuda:
                data.EN = data.EN.cuda(); trg_input = trg_input.cuda(); labels = labels.cuda()
                src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
            outputs = net(data.EN, trg_input, src_mask, trg_mask)
            outputs = outputs.view(-1, outputs.size(-1))
            acc += evaluate(outputs, labels)
    return acc/(i + 1)

def decode_outputs(outputs, labels, trg_vocab_obj):
    l = list(labels[:50].cpu().numpy())
    if outputs.is_cuda:
        o = list(torch.softmax(outputs, dim=1).max(1)[1][:50].cpu().numpy())
    else:
        o = list(torch.softmax(outputs, dim=1).max(1)[1][:50].numpy())
    print("Sample Output: ", " ".join(trg_vocab_obj.vocab.itos[i] for i in o))
    print("Sample Label: ", " ".join(trg_vocab_obj.vocab.itos[i] for i in l))
    