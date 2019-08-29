# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:06:29 2019

@author: tsd
"""
import os
import pickle
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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

def get_bpe_punc_mappings(vocab, punc_list=["!", "?", ".", ",", ":", ";",'\'']):
    mappings = {}
    for punc in punc_list:
        if punc in vocab.word_vocab.keys():
            mappings[punc] = vocab.word_vocab[punc]
        elif punc in vocab.bpe_vocab.keys():
            mappings[punc] = vocab.bpe_vocab[punc]
    return mappings

def get_punc_idx_mappings(mappings):
    idx_mappings = {v:k for k, v in enumerate(mappings.values())}
    return idx_mappings

def get_punc_idx_labels(tokens, idx_mappings):
    tokens1 = []
    idxs = idx_mappings.keys() # bpe_ids for punctuations
    for token in tokens:
        if token not in idxs:
            tokens1.append(1)
        else:
            tokens1.append(token)
    return tokens1

def remove_punc(tokens, mappings):
    '''
    tokens = bpe_tokenized ids; mappings = bpe_punctuation dictionary
    '''
    punc_idxs = mappings.values()
    tokens1 = []
    for token in tokens:
        if token not in punc_idxs:
            tokens1.append(token)
    return tokens1

def create_datasets(args):
    logger.info("Reading sentences corpus...")
    data_path = args.data_path
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
        mappings = get_bpe_punc_mappings(encoder)
        idx_mappings = get_punc_idx_mappings(mappings)
        
        save_as_pickle("./data/eng.pkl",\
                       df)
        encoder.save("./data/vocab.pkl")
        save_as_pickle("./data/mappings.pkl", mappings)
        save_as_pickle("./data/idx_mappings.pkl", idx_mappings)
        
        df.loc[:, 'train'] = df.progress_apply(lambda x: remove_punc(x['labels'], mappings), axis=1)
    return df

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

class punc_datasets(Dataset):
    def __init__(self, df):
        self.X = df['train']
        self.y1 = df['labels']
        self.y2 = df['labels_p']
        self.max_features_len = int(max(df['train'].apply(lambda x: len(x))))
        self.max_output_len = int(max(df['labels'].apply(lambda x: len(x))))
        
        def x_padder(x, max_len):
            if len(x) < max_len:
                x = np.array(x, dtype=int)
                x = np.append(x, np.ones((max_len-x.shape[-1]), dtype=int), axis=0)
                x = list(x)
            return x
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X.iloc[idx])
        y1 = torch.tensor(self.y1.iloc[idx])
        y2 = torch.tensor(self.y2.iloc[idx])
        return X, y1, y2

def get_TED_transcripts(args):
    logger.info("Collating TED transcripts...")
    with open(args.data_path, 'r', encoding='utf-8') as f: # "./data/train.tags.en-fr.en"
        text = f.read()

    soup = BeautifulSoup(text)
    results = soup.find_all("transcript")
    transcripts = []
    for result in results:
        transcripts.append(result.text)
    df = pd.DataFrame({'transcripts':transcripts})
    return df

def create_TED_datasets(args):
    df = get_TED_transcripts(args)
    sents = []
    logger.info("Splitting transcripts into sentences..")
    for transcript in tqdm(df['transcripts']):
        sents.extend(transcript.split('\n'))
    df = pd.DataFrame({'sents':sents})
    df.loc[df['sents'] == '', 'sents'] = None
    df.dropna(inplace=True) # remove blank rows
    df.loc[:, 'sents'] = df.progress_apply(lambda x: x['sents'].lower(), axis=1) # lower case
    
    if args.level == 'bpe':
        tokenizer_en = tokener("en")
        encoder = Encoder(vocab_size=args.bpe_vocab_size, pct_bpe=args.bpe_word_ratio, \
                          word_tokenizer=tokenizer_en.tokenize)
        
        logger.info("Training bpe, this might take a while...")
        text_list = list(df["sents"])
        encoder.fit(text_list); del text_list
        df.loc[:, 'labels'] = df.progress_apply(lambda x: next(encoder.transform([x["sents"]])), axis=1)
        mappings = get_bpe_punc_mappings(encoder)
        df.loc[:, 'train'] = df.progress_apply(lambda x: remove_punc(x['labels'], mappings), axis=1)
        idx_mappings = get_punc_idx_mappings(mappings)
        df.loc[:, 'labels_p'] = df.progress_apply(lambda x: get_punc_idx_labels(x['labels'], idx_mappings), axis=1)
        
        save_as_pickle("./data/eng.pkl",\
                       df)
        encoder.save("./data/vocab.pkl")
        save_as_pickle("./data/mappings.pkl", mappings)
        save_as_pickle("./data/idx_mappings.pkl", idx_mappings)
    
    return df
        
def load_dataloaders(args):
    if not os.path.isfile("./data/eng.pkl"):
        df = create_TED_datasets(args)
    else:
        df = load_pickle("./data/eng.pkl")
        logger.info("Loaded preprocessed data from file...")
    trainset = punc_datasets(df)
    train_length = len(trainset)
    max_features_len = trainset.max_features_len
    max_output_len = trainset.max_output_len
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,\
                              num_workers=0, collate_fn=Pad_Sequence(), pin_memory=False)
    return df, train_loader, train_length, max_features_len, max_output_len     