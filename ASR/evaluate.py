# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:51:35 2019

@author: WT
"""

import os
import torch
from .preprocessing import load_dataloaders
from .models.Transformer.transformer_model import SpeechTransformer, create_masks, create_gaussian_mask, create_window_mask
from models.LAS.LAS_model import LAS
from .train import load_model_and_optimizer
from .utils import load_pickle
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')


if __name__ == "__main__":
    args = load_pickle("args.pkl")
    args.batch_size = 1
    
    logger.info("Loading data...")
    train_loader, train_length, max_features_length, max_seq_length = load_dataloaders(args, folder="train-clean-5")
    vocab = load_pickle("vocab.pkl")
    
    logger.info("Loading model and optimizers...")
    cuda = torch.cuda.is_available()
    net, _, _, _, start_epoch, acc, g_mask1, g_mask2 = load_model_and_optimizer(args, vocab, \
                                                                                max_features_length, \
                                                                                max_seq_length, cuda)
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if args.model_no == 0:
                src_input, trg_input, f_len = data[0], data[1][:, :-1], data[2]
                labels = data[1][:,1:].contiguous().view(-1)
                src_mask, trg_mask = create_masks(src_input, trg_input[:,0].unsqueeze(0), f_len, args)
                if cuda:
                    src_input = src_input.cuda().float(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
                outputs = net(src_input, trg_input[:,0].unsqueeze(0), src_mask, trg_mask, g_mask1, g_mask2, infer=True)
                
            elif args.model_no == 1:
                src_input, trg_input = data[0], data[1][:, :-1]
                labels = data[1][:,1:].contiguous().view(-1)
                if cuda:
                    src_input = src_input.cuda().float(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                outputs = net(src_input, trg_input[:,0].unsqueeze(0), infer=True)
            if cuda:
                outputs = outputs.cpu().numpy(); labels = labels.cpu().numpy()
            else:
                outputs = outputs.numpy(); labels = labels.numpy()
            
            translated = vocab.convert_idx2w(outputs[0])
            ground_truth = vocab.convert_idx2w(labels)
            print("Translated: ")
            print("".join(w for w in translated if w not in ["<eos>", "<pad>"]))
            print("Ground truth: ")
            print("".join(w for w in ground_truth if w not in ["<eos>", "<pad>"]))
            break