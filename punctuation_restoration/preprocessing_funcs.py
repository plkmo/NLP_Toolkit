# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:06:29 2019

@author: tsd
"""
import os
import pickle
import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from .utils.word_char_level_vocab import tokener
from .utils.bpe_vocab import Encoder
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = filename
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = filename
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def create_trg_seq(tokens):
    '''
    input tokens: tokens tokenized from sentence
    '''
    punc_list = ["!", "?", ".", ",", ":", ";",]
    tokens1 = []
    for token in tokens:
        if token not in punc_list:
            tokens1.append(token)
    return tokens1

def create_labels(sent, tokenizer):
    punc_list = ["!", "?", ".", ",", ":", ";",]
    tokens = tokenizer.tokenize(sent)
    l = len(tokens)
    tokens1 = []
    for idx, token in enumerate(tokens):
        if token not in punc_list:
            if idx + 1 < l:
                if tokens[idx + 1] not in punc_list:
                    tokens1.append(token); tokens1.append(" ")
                else:
                    tokens1.append(token)
            else:
                tokens1.append(token)
        else:
            tokens1.append(token)
    return tokens1

def create_labels2(tokens):
    punc_list = ["!", "?", ".", ",", ":", ";",]
    tokens1 = []
    for token in tokens:
        if token not in punc_list:
            tokens1.append(" ")
        else:
            tokens1.append(token)
    return tokens1

def get_bpe_punc_mappings(vocab, punc_list=["!", "?", ".", ",", ":", ";",]):
    mappings = {}
    for punc in punc_list:
        if punc in vocab.word_vocab.keys():
            mappings[punc] = vocab.word_vocab[punc]
        elif punc in vocab.bpe_vocab.keys():
            mappings[punc] = vocab.bpe_vocab[punc]
    return mappings

def create_datasets(args):
    logger.info("Reading sentences corpus...")
    data_path = "./data/sentences.csv"
    df = pd.read_csv(data_path, names=["eng"])
    
    logger.info("Getting English sentences...")
    df['flag'] = df['eng'].apply(lambda x: re.search('\teng\t', x))
    df = df[df['flag'].notnull()] # filter out only english sentences
    df['eng'] = df['eng'].apply(lambda x: re.split('\teng\t', x)[1])
    df.drop(['flag'], axis=1, inplace=True)
    
    logger.info("Generaing train, labels...")
    if args.level == "word":
        tokenizer_en = tokener("en")
        df["labels"] = df.progress_apply(lambda x: create_labels(x["eng"], tokenizer_en), axis=1)
        df["labels2"] = df.progress_apply(lambda x: create_labels2(x["labels"]), axis=1)
        df["train"] = df.progress_apply(lambda x: create_trg_seq(x["labels"]), axis=1)
        save_as_pickle("./data/eng.pkl",\
                       df)
        
    elif args.level == "bpe":
        tokenizer_en = tokener("en")
        encoder = Encoder(vocab_size=args.bpe_vocab_size, pct_bpe=args.bpe_word_ratio, \
                          word_tokenizer=tokenizer_en.tokenize)
        
        logger.info("Training bpe, this might take a while...")
        text_list = list(df["eng"])
        encoder.fit(text_list); del text_list
        df.loc[:, 'labels'] = df.progress_apply(lambda x: next(encoder.transform([x["eng"]])), axis=1)
        save_as_pickle("./data/eng.pkl",\
                       df)
        encoder.save("./data/vocab.pkl")
    return df

class punc_datasets(Dataset):
    def __init__(self, df):
        self.X = df['train']
        self.y1 = df['labels']
        self.y2 = df['labels2']
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X.iloc[idx])
        y1 = torch.tensor(self.y1.iloc[idx])
        y2 = torch.tensor(self.y2.iloc[idx])
        return X, y1, y2

def load_dataloaders(args):
    if not os.path.isfile("./data/eng.pkl"):
        df = create_datasets(args)
    else:
        df = load_pickle("./data/eng.pkl")
    trainset = punc_datasets(df)
    train_length = len(trainset)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,\
                              num_workers=0, pin_memory=False)
    return df, train_loader, train_length