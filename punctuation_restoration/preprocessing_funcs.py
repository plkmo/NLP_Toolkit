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
    input tokens: tokens tokenized from sentence ["!", "?", ".", ",", ":", ";",]
    '''
    punc_list = ["!", "?", ".", ","]
    tokens1 = []
    for token in tokens:
        if token not in punc_list:
            tokens1.append(token)
    return tokens1

def create_labels(sent, tokenizer):
    punc_list = ["!", "?", ".", ","]
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
    punc_list = ["!", "?", ".", ","]
    tokens1 = []
    for token in tokens:
        if token not in punc_list:
            tokens1.append(" ")
        else:
            tokens1.append(token)
    return tokens1

def get_bpe_punc_mappings(vocab, punc_list=["!", "?", ".", ","]): # ["!", "?", ".", ",", ":", ";",'\'']
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
    tokens1 = []; punc_locs = []; no_tokens = 1
    idxs = idx_mappings.keys() # bpe_ids for punctuations
    for idx, token in enumerate(tokens):
        if token not in idxs:
            tokens1.append(1)
        else:
            tokens1.append(token)
            punc_locs.append(idx)
            no_tokens = 0
            
    if idx not in punc_locs:
        no_tokens = 1
        punc_locs.append(idx)
    #if len(punc_locs) > 0:
    tokens2 = []; x = 1
    for i in punc_locs:
        tokens2 += tokens1[x:(i + 1)]
        x = i + 2
    tokens2 = [idx_mappings[t] if t in idxs else len(idxs) for t in tokens2]
    if no_tokens == 1:
        tokens2.append(len(idxs))
    '''
    c = 0
    while c < (len(tokens) - len(punc_locs) - 1):
        tokens2.append(1); c = len(tokens2)
    tokens2 = [idx_mappings[t] if t in idxs else len(idxs) for t in tokens2]
    else:
    tokens2 = [len(idxs) for _ in range(len(tokens))]
    '''
    return tokens2

def get_labels2(tokens, idx_mappings):
    punc_idxs = idx_mappings.keys()
    tokens1 = []
    for token in tokens:
        if token not in punc_idxs:
            tokens1.append(idx_mappings['word'])
        else:
            tokens1.append(idx_mappings[token])
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

def pad_sos_eos(x, sos, eos):
    return [sos] + x + [eos] 

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, label_pad_value=1, label2_pad_value=7):
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.label_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])
        return seqs_padded, labels_padded, labels2_padded, x_lengths, y_lengths, y2_lengths

class punc_datasets(Dataset):
    def __init__(self, df, label_pad_value=1, label2_pad_value=7, labels2=True, pad_max_length=False):
        self.pad_max_length = pad_max_length
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        self.X = df['train']
        self.y1 = df['labels']
        if labels2:
            self.y2 = df['labels2']
        else:
            self.y2 = df['labels_p']
        self.max_features_len = int(max(df['train'].apply(lambda x: len(x))))
        self.max_output_len = int(max(df['labels'].apply(lambda x: len(x))))
        
        def x_padder(x, max_len):
            if len(x) < max_len:
                x = np.array(x, dtype=int)
                x = np.append(x, np.ones((max_len-x.shape[-1]), dtype=int), axis=0)
                x = list(x)
            return x
        
        if pad_max_length:
            logger.info("Padding sequences to max_lengths %d, %d" % (80, self.max_output_len))
            self.X = df.progress_apply(lambda x: x_padder(x['train'], 80), axis=1)
        
    
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
        df.loc[:, 'length'] = df.progress_apply(lambda x: len(x['labels']), axis=1)
        df = df[df['length'] > 3] # filter out too short sentences
            
        mappings = get_bpe_punc_mappings(encoder)
        df.loc[:, 'train'] = df.progress_apply(lambda x: remove_punc(x['labels'], mappings), axis=1)
        idx_mappings = get_punc_idx_mappings(mappings)
        df.loc[:, 'labels_p'] = df.progress_apply(lambda x: get_punc_idx_labels(x['labels'], idx_mappings), axis=1)
        
        logger.info("Padding sos, eos tokens...") 
        idx_mappings['word'] = len(idx_mappings) # 4 = word
        idx_mappings['sos'] = len(idx_mappings)  # 5
        idx_mappings['eos'] = len(idx_mappings)  # 6
        idx_mappings['pad'] = len(idx_mappings)  # 7 = pad
        df.loc[:, 'labels2'] = df.progress_apply(lambda x: get_labels2(x['labels'], idx_mappings), axis=1)

        df.loc[:, 'labels'] = df.progress_apply(lambda x: pad_sos_eos(x["labels"], encoder.word_vocab["__sos"], \
                                                      encoder.word_vocab["__eos"]), axis=1) # pad sos eos
        df.loc[:, 'labels_p'] = df.progress_apply(lambda x: pad_sos_eos(x["labels_p"], idx_mappings['sos'], \
                                                      idx_mappings['eos']), axis=1)
        df.loc[:, 'labels2'] = df.progress_apply(lambda x: pad_sos_eos(x["labels2"], idx_mappings['sos'], \
                                                      idx_mappings['eos']), axis=1)
        df.loc[:, 'labels_p_length'] = df.progress_apply(lambda x: len(x['labels_p']), axis=1)
        df.loc[:, 'labels2_length'] = df.progress_apply(lambda x: len(x['labels2']), axis=1)
        
        logger.info("Limiting tokens to max_encoder_length...")
        df = df[df['length'] <= (args.max_encoder_len - 2)]
        
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
    vocab = Encoder.load("./data/vocab.pkl")
    idx_mappings = load_pickle("./data/idx_mappings.pkl") # {250: 0, 34: 1, 5: 2, 4: 3, 'word': 4, 'sos': 5, 'eos': 6, 'pad': 7}
    
    if args.model_no == 0:
        trainset = punc_datasets(df=df, label_pad_value=vocab.word_vocab['__pad'], label2_pad_value=idx_mappings['pad'],\
                                 pad_max_length=False)
    elif args.model_no == 1:
        trainset = punc_datasets(df=df, label_pad_value=vocab.word_vocab['__pad'], label2_pad_value=idx_mappings['pad'],\
                                 pad_max_length=True)
    
    train_length = len(trainset)
    max_features_len = trainset.max_features_len
    max_output_len = trainset.max_output_len
    
    if args.model_no == 0:
        PS = Pad_Sequence(label_pad_value=vocab.word_vocab['__pad'], label2_pad_value=idx_mappings['pad'])
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,\
                                  num_workers=0, collate_fn=PS, pin_memory=False)
    elif args.model_no == 1:
        PS = Pad_Sequence(label_pad_value=vocab.word_vocab['__pad'], label2_pad_value=idx_mappings['pad'])
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,\
                                  num_workers=0, collate_fn=PS, pin_memory=False)
    return df, train_loader, train_length, max_features_len, max_output_len