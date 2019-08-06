# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:23:19 2019

@author: WT
"""

import pandas as pd
import os
import re
import spacy
import logging

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

def tokenize_data():
    logger.info("Loading raw data and tokenizing...")
    df = pd.read_csv(os.path.join("./data/", "english.txt"), names=["English"])
    df["French"] = pd.read_csv(os.path.join("./data/", "french.txt"), names=["French"])["French"]    
    tokenizer_fr = tokener("fr")
    tokenizer_en = tokener("en")
    df["English"] = df["English"].apply(lambda x: tokenizer_en.tokenize(x))
    df["French"] = df["French"].apply(lambda x: tokenizer_fr.tokenize(x))
    df.to_csv(os.path.join("./data/", "df.csv"), index=False)
    logger.info("Done loading raw data and tokenizing!")
    
if __name__ == "__main__":
    tokenize_data()