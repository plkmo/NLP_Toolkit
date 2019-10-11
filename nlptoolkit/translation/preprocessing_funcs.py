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
from spacy.lang.zh import Chinese
from tqdm import tqdm
import logging

tqdm.pandas(desc="Progress_bar")
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
        d = {"en":"en_core_web_lg", "fr":"fr_core_news_sm"}
        self.lang = lang
        if lang in ['fr', 'en']:
            self.ob = spacy.load(d[lang])
        elif lang == 'zh':
            self.ob = ch_tokener() 
    
    def tokenize(self, sent):
        sent = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sent))
        sent = re.sub(r"\!+", "!", sent)
        sent = re.sub(r"\,+", ",", sent)
        sent = re.sub(r"\?+", "?", sent)
        sent = re.sub(r"[ ]+", " ", sent)
        
        if self.lang in ['en', 'fr']:
            sent = sent.lower()
            sent = [token.text for token in self.ob.tokenizer(sent) if token.text != " "]
            sent = " ".join(sent)
        elif self.lang == 'zh':
            sent = self.ob.tokenize(sent)
        return sent
    
class ch_tokener(object):
    def __init__(self):
        self.nlp = Chinese()
    
    def tokenize(self, sent):
        doc = self.nlp(sent)
        sent = " ".join(str(token) for token in doc)
        return sent

def dum_tokenizer(sent):
    return sent.split()

def tokenize_data(args):
    logger.info("Loading raw data and tokenizing...")
    with open(args.src_path, "r", encoding="utf8") as f:
        eng_text = f.read()
    with open(args.trg_path, "r", encoding="utf8") as f:
        ch_text = f.read()
    eng = re.split("[\n]+", eng_text)
    ch = re.split("[\n]+", ch_text)
    eng_list, ch_list = [], []
    
    assert len(eng) == len(ch)
    for e, c in tqdm(zip(eng, ch), total=len(eng)):
        if (len(e) != 0) and (len(c) != 0):
            eng_list.append(e); ch_list.append(c)
    df = pd.DataFrame(data={"English":eng_list, "French":ch_list})
    '''
    df = pd.read_csv(args.src_path, names=["English"])
    df["French"] = pd.read_csv(args.trg_path, names=["French"])["French"]
    '''
    tokenizer_fr = tokener(args.trg_lang)
    tokenizer_en = tokener(args.src_lang)
    df["English"] = df.progress_apply(lambda x: tokenizer_en.tokenize(x["English"]), axis=1)
    df["French"] = df.progress_apply(lambda x: tokenizer_fr.tokenize(x["French"]), axis=1)
    df['eng_len'] = df.progress_apply(lambda x: len(x['English']), axis=1)
    df['fr_len'] = df.progress_apply(lambda x: len(x['French']), axis=1)
    max_len = max(args.max_encoder_len, args.max_decoder_len)
    df = df[(df['fr_len'] <= max_len) & (df['eng_len'] <= max_len)] # limiting to max length
    
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