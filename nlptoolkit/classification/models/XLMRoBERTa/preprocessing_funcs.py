# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:44:05 2019

@author: WT
"""
import os
import pickle
import pandas as pd
from .tokenization_xlm_roberta import XLMRobertaTokenizer
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

'''
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
input_ids = torch.tensor(tokenizer.encode("Hello how are you?"))
print(tokenizer.decode(input_ids.numpy().tolist()))
'''

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
### remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        token = token.lower()
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1

def dummy_fun(doc):
    return doc

def preprocess(args):
    logger.info("Preprocessing data...")
    df_train = pd.read_csv(args.train_data)
    df_test = pd.read_csv(args.infer_data)
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=False)
    tokens_length = args.tokens_length # max tokens length
    
    logger.info("Tokenizing data...")
    ### tokenize data for ALBERT
    df_train.loc[:, "text"] = df_train["text"].apply(lambda x: tokenizer.tokenize(x))
    df_train.loc[:, "text"] = df_train["text"].apply(lambda x: tokenizer.convert_tokens_to_ids(x[:(tokens_length-1)]))
    df_test.loc[:, "text"] = df_test["text"].apply(lambda x: tokenizer.tokenize(x))
    df_test.loc[:, "text"] = df_test["text"].apply(lambda x: tokenizer.convert_tokens_to_ids(x[:(tokens_length-1)]))
    
    ### fill up reviews with [PAD] if word length less than tokens_length
    def filler(x, pad=0, length=tokens_length):
        dum = x
        while (len(dum) < length):
            dum.append(pad)
        return dum
    
    logger.info("Padding sequences...")
    df_train.loc[:, "text"] = df_train["text"].apply(lambda x: filler(x))
    df_test.loc[:, "text"] = df_test["text"].apply(lambda x: filler(x))
    df_train.loc[:, "fills"] = df_train["text"].apply(lambda x: x.count(0))
    df_test.loc[:, "fills"] = df_test["text"].apply(lambda x: x.count(0))
    
    logger.info("Saving..")
    df_train.to_pickle(os.path.join("./data/", "train_processed.pkl"))
    df_test.to_pickle(os.path.join("./data/", "infer_processed.pkl"))
    logger.info("Done!")
    
