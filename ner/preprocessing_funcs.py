# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:25:58 2019

@author: WT
"""

import os
import pandas as pd
import csv
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import logging
from .utils.misc_utils import save_as_pickle, load_pickle
from .utils.word_char_level_vocab import tokener, vocab
from .utils.bpe_vocab import Encoder

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def pad_sos_eos(x, sos, eos):
    return [sos] + x + [eos] 

class args():
    def __init__(self):
        self.batch_size = 5

def clean_and_tokenize_text(text, table, tokenizer, clean_only=False):
    if isinstance(text, str):
        text = text.replace("(CNN) -- ","").replace("U.N.", "UN").replace("U.S.", "USA")
        text = text.replace(".", ". ").replace(",", ", ").replace("?", "? ").replace("!", "! ")
        text = text.translate(table)
        if clean_only == False:
            text = tokenizer.tokenize(text)
            text = [w for w in text if not any(char.isdigit() for char in w)]
        return text

def get_NER_data(args, load_extracted=True):
    """
    Extracts NER dataset, saves then
    returns dataframe containing body (main text) and NER tags columns
    table: table containing symbols to remove from text
    tokenizer: tokenizer to tokenize text into word tokens
    """
    train_path = args.train_path
    if args.test_path is not None:
        test_path = args.test_path
    else:
        test_path = None
        
    table = str.maketrans("", "", '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~')
    if load_extracted:
        df_train =  load_pickle("df_train.pkl")
        if os.path.isfile("./data/df_test.pkl") is not None:
            df_test = load_pickle("df_test.pkl")
            
    else:
        logger.info("Extracting data stories...")
        with open(train_path, "r", encoding="utf8") as f:
            text = f.readlines()
        
    
    return text

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=1)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=1)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        return seqs_padded, labels_padded, x_lengths, y_lengths

class text_dataset(Dataset):
    def __init__(self, df, args):
        
        def x_padder(x, max_len):
            if len(x) < max_len:
                x = np.array(x, dtype=int)
                x = np.append(x, np.ones((max_len-x.shape[-1]), dtype=int), axis=0)
                x = list(x)
            return x
        
        if args.model_no == 1:
            self.X = df["body"].apply(lambda x: x_padder(x, args.max_features_length))
        else:
            self.X = df["body"]
        self.y = df["highlights"]
        self.max_x_len = int(max(df["body"].apply(lambda x: len(x))))
        self.max_y_len = int(max(df["highlights"].apply(lambda x: len(x))))
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X.iloc[idx])
        y = torch.tensor(self.y.iloc[idx])
        return X, y

def load_dataloaders(args):
    """Load processed data if exist, else do preprocessing and loads it.  Feeds preprocessed data into dataloader, 
    returns dataloader """
    logger.info("Loading dataloaders...")
    p_path = os.path.join("./data/", "df_unencoded_CNN.pkl")
    train_path = os.path.join("./data/", "df_encoded_CNN.pkl")
    if (not os.path.isfile(p_path)) and (not os.path.isfile(train_path)):
        df = get_CNN_data(args, load_extracted=False)
    elif os.path.isfile(p_path) and (not os.path.isfile(train_path)):
        df = get_CNN_data(args, load_extracted=True)
    elif os.path.isfile(train_path):
        df = load_pickle("df_encoded_CNN.pkl")
    
    trainset = text_dataset(df, args)
    max_features_length = trainset.max_x_len
    max_seq_len = trainset.max_y_len
    train_length = len(trainset)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,\
                              num_workers=0, collate_fn=Pad_Sequence(), pin_memory=False)
    return train_loader, train_length, max_features_length, max_seq_len
