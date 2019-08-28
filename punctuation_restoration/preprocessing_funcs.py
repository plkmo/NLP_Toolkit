# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:06:29 2019

@author: tsd
"""
import os
import pickle
import pandas as pd
import spacy
import re
from torch.utils.data import Dataset, DataLoader
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

class tokener(object):
    def __init__(self, lang):
        d = {"en":"en_core_web_lg", "fr":"fr_core_news_sm"} # "en_core_web_lg"
        self.ob = spacy.load(d[lang])
    
    def tokenize(self, sent):
        sent = re.sub(r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|]", " ", str(sent))
        #sent = re.sub(r"\!+", "!", sent)
        #sent = re.sub(r"\,+", ",", sent)
        #sent = re.sub(r"\?+", "?", sent)
        sent = re.sub(r"[ ]+", " ", sent)
        sent = sent.lower()
        sent = [token.text for token in self.ob.tokenizer(sent) if token.text != " "]
        #sent = " ".join(sent)
        return sent

def remove_punct(tokens):
    punc_list = ["!", "?", ".", ",", ":", ";",]
    tokens1 = []
    for token in tokens:
        if token not in punc_list:
            tokens1.append(token)
    return tokens1

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

def create_datasets():
    logger.info("Reading sentences corpus...")
    data_path = "C:/Users/tsd/Desktop/Python_Projects/DSL/Repositories/NLP_Toolkit/data/Sentences_tatoeba/sentences.csv"
    df = pd.read_csv(data_path, names=["eng"])
    
    logger.info("Getting English sentences...")
    df['flag'] = df['eng'].apply(lambda x: re.search('\teng\t', x))
    df = df[df['flag'].notnull()] # filter out only english sentences
    df['eng'] = df['eng'].apply(lambda x: re.split('\teng\t', x)[1])
    df.drop(['flag'], axis=1, inplace=True)
    
    logger.info("Generaing train, labels...")
    tokenizer_en = tokener("en")
    df["labels"] = df.progress_apply(lambda x: create_labels(x["eng"], tokenizer_en), axis=1)
    df["labels2"] = df.progress_apply(lambda x: create_labels2(x["labels"]), axis=1)
    df["train"] = df.progress_apply(lambda x: create_trg_seq(x["labels"]), axis=1)
    save_as_pickle("C:/Users/tsd/Desktop/Python_Projects/DSL/Repositories/NLP_Toolkit/data/Sentences_tatoeba/eng.pkl",\
                   df)
    return df

class punc_datasets(Dataset):
    def __init__(self, df):
        self.X = df['train']
        self.y1 = df['labels']
        self.y2 = df['labels2']
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X.iloc[idx], self.y1.iloc[idx], self.y2.iloc[idx]

def load_dataloaders(args):
    if not os.path.isfile("C:/Users/tsd/Desktop/Python_Projects/DSL/Repositories/NLP_Toolkit/data/Sentences_tatoeba/eng.pkl"):
        df = create_datasets()
    else:
        df = load_pickle("C:/Users/tsd/Desktop/Python_Projects/DSL/Repositories/NLP_Toolkit/data/Sentences_tatoeba/eng.pkl")
    trainset = punc_datasets(df)
    train_length = len(trainset)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,\
                              num_workers=0, pin_memory=False)
    return train_loader, train_length
        