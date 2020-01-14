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

def get_data(args, load_extracted=True):
    """
    Extracts CNN and/or dailymain dataset, saves then
    returns dataframe containing body (main text) and highlights (summarized text)
    table: table containing symbols to remove from text
    tokenizer: tokenizer to tokenize text into word tokens
    """
    path = args.data_path1
    tokenizer_en = tokener()
    table = str.maketrans("", "", '"#$%&\'()*+-/:;<=>@[\\]^_`{|}~')
    if load_extracted:
        df = load_pickle("df_unencoded.pkl")
    else:
        logger.info("Extracting CNN stories...")
        df = pd.DataFrame(index=[i for i in range(len(os.listdir(path)))], columns=["body", "highlights"])
        for idx, file in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path))):
            with open(os.path.join(path, file), encoding="utf8") as csv_file:
                csv_reader = csv.reader(csv_file)
                text = ""
                for row in csv_reader:
                    text += "".join(t for t in row)
            highlights = re.search("@highlight(.*)", text).group(1)
            highlights = highlights.replace("@highlight", ". ")
            body = text[:re.search("@highlight", text).span(0)[0]]
            df.iloc[idx]["body"] = body
            df.iloc[idx]["highlights"] = highlights
            
        if len(args.data_path2) > 2:
            path = args.data_path2
            logger.info("Extracting dailymail stories...")
            df1 = pd.DataFrame(index=[i for i in range(len(os.listdir(path)))], columns=["body", "highlights"])
            for idx, file in tqdm(enumerate(os.listdir(path)), total=len(os.listdir(path))):
                with open(os.path.join(path, file), encoding="utf8") as csv_file:
                    csv_reader = csv.reader(csv_file)
                    text = ""
                    for row in csv_reader:
                        text += "".join(t for t in row)
                highlights = re.search("@highlight(.*)", text).group(1)
                highlights = highlights.replace("@highlight", ". ")
                body = text[:re.search("@highlight", text).span(0)[0]]
                df1.iloc[idx]["body"] = body
                df1.iloc[idx]["highlights"] = highlights
            df = pd.concat([df, df1], ignore_index=True)
            del df1
        
        save_as_pickle("df_unencoded.pkl", df)
    logger.info("Dataset length: %d" % len(df))    
    
    if (args.level == "word") or (args.level == "char"):
        logger.info("Tokenizing and cleaning extracted text...")
        df.loc[:, "body"] = df.apply(lambda x: clean_and_tokenize_text(x["body"], table, tokenizer_en), axis=1)
        df.loc[:, "highlights"] = df.apply(lambda x: clean_and_tokenize_text(x["highlights"], table, tokenizer_en), \
                                      axis=1)
        df.loc[:, "body_length"] = df.apply(lambda x: len(x['body']), axis=1)
        df.loc[:, "highlights_length"] = df.apply(lambda x: len(x['highlights']), axis=1)
        df = df[(df["body_length"] > 0) & (df["highlights_length"] > 0)]
                
        logger.info("Limiting to max features length, building vocab and converting to id tokens...")
        df = df[df["body_length"] <= args.max_features_length]
        v = vocab(level=args.level)
        v.build_vocab(df["body"])
        v.build_vocab(df["highlights"])
        df.loc[:, "body"] = df.apply(lambda x: v.convert_w2idx(x["body"]), axis=1)
        df.loc[:, "highlights"] = df.apply(lambda x: v.convert_w2idx(x["highlights"]), axis=1)
        df.loc[:, "highlights"] = df.apply(lambda x: pad_sos_eos(x["highlights"], 0, 2), axis=1)
        save_as_pickle("df_encoded.pkl", df)
        save_as_pickle("vocab.pkl", v)
        
    elif args.level == "bpe":
        encoder = Encoder(vocab_size=args.bpe_vocab_size, pct_bpe=args.bpe_word_ratio, word_tokenizer=tokenizer_en.tokenize)
        df.loc[:, "body"] = df.apply(lambda x: clean_and_tokenize_text(x["body"], table, tokenizer_en, clean_only=True), axis=1)
        df.loc[:, "highlights"] = df.apply(lambda x: clean_and_tokenize_text(x["highlights"], table, tokenizer_en, clean_only=True), \
                                      axis=1)
        logger.info("Training bpe, this might take a while...")
        text_list = list(df["body"])
        text_list.extend(list(df["highlights"]))
        encoder.fit(text_list); del text_list
        
        logger.info("Tokenizing to ids and limiting to max features length...")
        df.loc[:, "body"] = df.apply(lambda x: next(encoder.transform([x["body"]])), axis=1)
        df.loc[:, "highlights"] = df.apply(lambda x: next(encoder.transform([x["highlights"]])), axis=1)
        df.loc[:, "body_length"] = df.apply(lambda x: len(x['body']), axis=1)
        df.loc[:, "highlights_length"] = df.apply(lambda x: len(x['highlights']), axis=1)
        df = df[(df["body_length"] > 0) & (df["highlights_length"] > 0)]
        df = df[df["body_length"] <= args.max_features_length]
        
        '''
        logger.info("Converting tokens to ids...")
        df.loc[:, "body"] = df.apply(lambda x: next(encoder.transform(list(" ".join(t for t in x["body"])))),\
                                                  axis=1)
        df.loc[:, "highlights"] = df.apply(lambda x: next(encoder.transform(list(" ".join(t for t in x["highlights"])))),\
                                              axis=1)
        '''
        df.loc[:, "highlights"] = df.apply(lambda x: pad_sos_eos(x["highlights"], encoder.word_vocab["__sos"], encoder.word_vocab["__eos"]),\
                                              axis=1)
        
        save_as_pickle("df_encoded.pkl", df)
        encoder.save("./data/vocab.pkl")
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
    p_path = os.path.join("./data/", "df_unencoded.pkl")
    train_path = os.path.join("./data/", "df_encoded.pkl")
    if (not os.path.isfile(p_path)) and (not os.path.isfile(train_path)):
        df = get_data(args, load_extracted=False)
    elif os.path.isfile(p_path) and (not os.path.isfile(train_path)):
        df = get_data(args, load_extracted=True)
    elif os.path.isfile(train_path):
        df = load_pickle("df_encoded.pkl")
    
    # Train-Test split
    msk = np.random.rand(len(df)) < args.train_test_ratio
    trainset = df[msk]
    testset = df[~msk]
    
    trainset = text_dataset(trainset, args)
    max_features_length = trainset.max_x_len
    max_seq_len = trainset.max_y_len
    train_length = len(trainset)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,\
                              num_workers=0, collate_fn=Pad_Sequence(), pin_memory=False)
    
    testset = text_dataset(testset, args)
    test_length = len(testset)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True,\
                              num_workers=0, collate_fn=Pad_Sequence(), pin_memory=False)
    return train_loader, train_length, max_features_length, max_seq_len, test_loader, test_length
