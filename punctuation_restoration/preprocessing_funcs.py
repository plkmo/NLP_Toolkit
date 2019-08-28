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
import logging

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
        d = {"en":"en_core_web_lg", "fr":"fr_core_news_sm"}
        self.ob = spacy.load(d[lang])
    
    def tokenize(self, sent):
        sent = re.sub(r"[\*\"\n\\…\+\-\/\=\(\)‘•€\[\]\|]", " ", str(sent))
        #sent = re.sub(r"\!+", "!", sent)
        #sent = re.sub(r"\,+", ",", sent)
        #sent = re.sub(r"\?+", "?", sent)
        sent = re.sub(r"[ ]+", " ", sent)
        sent = sent.lower()
        sent = [token.text for token in self.ob.tokenizer(sent) if token.text != " "]
        sent = []
        #sent = " ".join(sent)
        return sent

def remove_punct(tokens):
    punc_list = ["!", "?", ".", ",", ":", ";",]
    tokens1 = []
    for token in tokens:
        if token in punc_list:
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
    df["labels"] = df["eng"].apply(lambda x: tokenizer_en.tokenize(x))
    df["train"] = df["labels"].apply(lambda x: remove_punct(x))
    save_as_pickle("C:/Users/tsd/Desktop/Python_Projects/DSL/Repositories/NLP_Toolkit/data/Sentences_tatoeba/eng.pkl",\
                   df)
    return df

if __name__ == '__main__':
    if not os.path.isfile("C:/Users/tsd/Desktop/Python_Projects/DSL/Repositories/NLP_Toolkit/data/Sentences_tatoeba/eng.pkl"):
        df = create_datasets()
    else:
        df = load_pickle("C:/Users/tsd/Desktop/Python_Projects/DSL/Repositories/NLP_Toolkit/data/Sentences_tatoeba/eng.pkl")
    
    