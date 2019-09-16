# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 09:17:59 2019

@author: WT
"""
import torch
from torch.autograd import Variable
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_model_and_optimizer, decode_outputs
from .utils.misc_utils import load_pickle
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def infer(args, from_data=False):
    args.batch_size = 1
    cuda = torch.cuda.is_available()
    
    vocab = load_pickle("vocab.pkl")
        
    logger.info("NER Vocabulary size: %d" % len(vocab.ner2idx))    
    
    logger.info("Loading model and optimizers...")
    net, _, _, _, start_epoch, acc = load_model_and_optimizer(args, cuda)
    
    
    if from_data:
        train_loader, train_length, test_loader, test_length = load_dataloaders(args)
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                
                if args.model_no == 0:
                    src_input = data[0]
                    labels = data[1]
                    src_mask = (src_input != 0).float()
                    if cuda:
                        src_input = src_input.cuda().long(); labels = labels.cuda().long()
                        src_mask = src_mask.cuda()
                    outputs = net(src_input, attention_mask=src_mask)
                    outputs = outputs[0][:, 1:-1, :]
                    
                elif args.model_no == 1:
                    src_input, trg_input = data[0], data[1][:, :-1]
                    labels = data[1][:,1:].contiguous().view(-1)
                    if cuda:
                        src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    outputs = net(src_input, trg_input)
                
                #print(outputs.shape); print(labels.shape)
                #outputs = outputs.reshape(-1, outputs.size(-1))
                #outputs = outputs.view(-1, outputs.size(-1))
                
                decode_outputs(outputs, labels, vocab.idx2ner, args, reshaped=True)
                print("")
                time.sleep(7)
    else:
        pass