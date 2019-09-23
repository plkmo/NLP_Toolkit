# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:23:19 2019

@author: WT
"""
import sys
import csv
import pandas as pd
import os
import re
import torchtext
from torchtext.data import BucketIterator
import spacy
import logging

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
        
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

class tokener(object):
    def __init__(self, lang):
        d = {"en":"en_core_web_sm", "fr":"fr_core_news_sm"}
        self.ob = spacy.load(d[lang])
    
    def tokenize(self, sent):
        sent = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sent))
        sent = re.sub(r"\!+", "!", sent)
        sent = re.sub(r"\,+", ",", sent)
        sent = re.sub(r"\?+", "?", sent)
        sent = re.sub(r"[ ]+", " ", sent)
        sent = sent.lower()
        sent = [token.text for token in self.ob.tokenizer(sent) if token.text != " "]
        sent = " ".join(sent)
        return sent

def dum_tokenizer(sent):
    return sent.split()

def tokenize_data(args):
    logger.info("Loading raw data and tokenizing...")
    df = pd.read_csv(args.src_path, names=["English"])
    df["French"] = pd.read_csv(args.trg_path, names=["French"])["French"]    
    tokenizer_fr = tokener("fr")
    tokenizer_en = tokener("en")
    df["English"] = df["English"].apply(lambda x: tokenizer_en.tokenize(x))
    df["French"] = df["French"].apply(lambda x: tokenizer_fr.tokenize(x))
    df.to_csv(os.path.join("./data/", "df.csv"), index=False)
    logger.info("Done loading raw data and tokenizing!")
    
def load_dataloaders(args):
    logger.info("Preparing dataloaders...")
    FR = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, init_token="<sos>", eos_token="<eos>",\
                              batch_first=True)
    EN = torchtext.data.Field(tokenize=dum_tokenizer, lower=True, batch_first=True)
    
    train_path = os.path.join("./data/", "df.csv")
    if not os.path.isfile(train_path):
        tokenize_data(args)
    train = torchtext.data.TabularDataset(train_path, format="csv", \
                                             fields=[("EN", EN), ("FR", FR)])
    FR.build_vocab(train)
    EN.build_vocab(train)
    train_iter = BucketIterator(train, batch_size=args.batch_size, repeat=False, sort_key=lambda x: (len(x["EN"]), len(x["FR"])),\
                                shuffle=True, train=True)
    train_length = len(train)
    logger.info("Loaded dataloaders.")
    return train_iter, FR, EN, train_length