# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:51:35 2019

@author: WT
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from .preprocessing_funcs import extract_feature, padded_dataset
from .models.Transformer.transformer_model import create_masks
from .train_funcs import load_model_and_optimizer
from .utils import load_pickle
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')


def infer(file_path, speaker=None):
    args = load_pickle("args.pkl")
    args.batch_size = 1
    
    logger.info("Loading model and optimizers...")
    vocab = load_pickle("vocab.pkl")
    print("vocab size:", len(vocab.w2idx))
    cuda = torch.cuda.is_available()
    net, _, _, _, start_epoch, acc, g_mask1, g_mask2 = load_model_and_optimizer(args, vocab, \
                                                                                200, \
                                                                                200, cuda)
    #print(g_mask1.shape, g_mask2.shape)
    if args.model_no == 0:
        args.max_seq_len = net.max_decoder_len
    elif args.model_no == 1:
        args.max_seq_len = 100
    
    logger.info("Loading data...")
    features, original_len = extract_feature(file_path, args)
    df = pd.DataFrame(data=[(features, original_len)], columns=['features', 'features_len'])
    
    logger.info("Normalizing...")
    channel_mu = []; channel_std = []
    for channel in range(3):
        f_list = []
        for row, l in zip(df["features"], df["features_len"]):
            row = row[:,:,:int(l)]
            f_list.extend(list(row[channel].reshape(-1)))
        f_list = np.array(f_list)
        channel_mu.append(f_list.mean())
        channel_std.append(f_list.std())
        
    def speaker_norm(feature, stats):
        channel_mu, channel_std = stats
        for idx, (mu, std) in enumerate(zip(channel_mu, channel_std)):
            feature[idx, :, :] = (feature[idx, :, :] - mu)/std
        return feature
    
    if os.path.isfile("./data/speaker_stats.pkl") and (speaker != None):
        speaker_stats = load_pickle("speaker_stats.pkl")
        df["features"] = df.apply(lambda x: speaker_norm(x["features"], speaker_stats[speaker]), axis=1)
    else:
        df["features"] = df.apply(lambda x: speaker_norm(x["features"], (channel_mu, channel_std)), axis=1)
    
    inferset = padded_dataset(df, args, labels=False)
    infer_loader = DataLoader(inferset, batch_size=args.batch_size, shuffle=False, \
                              num_workers=0, pin_memory=False)
    
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(infer_loader):
            if args.model_no == 0:
                src_input, trg_input, f_len = data[0], data[1][:, :-1], data[2]
                labels = data[1][:,1:].contiguous().view(-1)
                src_mask, trg_mask = create_masks(src_input, trg_input, f_len, args)
                if cuda:
                    src_input = src_input.cuda().float(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
                print("Masks")
                print(src_input.shape, trg_input[:,0].unsqueeze(0).shape, src_mask.shape, trg_mask.shape, g_mask1.shape, g_mask2.shape)
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
