# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:35:16 2019

@author: WT
"""
from .misc_utils import save_as_pickle
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

class vocab_mapper(object):
    def __init__(self, df_train, df_test=None, ner_only=True):
        
        logger.info("Building vocab...")
        sents = []; ners = []
        for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
            sent, ner = row[0], row[1]
            sents.extend(sent); ners.extend(ner)
        sents = list(set(sents)); ners = list(set(ners))
        
        if df_test is not None:
            for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
                sent, ner = row[0], row[1]
                sents.extend(sent); ners.extend(ner)
            sents = list(set(sents)); ners = list(set(ners))
        
        self.word2idx = {k:v for v, k in enumerate(sents, 1)}
        self.word2idx['<pad>'] = 0
        self.ner2idx = {k:v for v, k in enumerate(ners, 1)}
        self.ner2idx.update({'<pad>':0}) #, '<sos>':1, '<eos>':2})
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        self.idx2ner = {v:k for k, v in self.ner2idx.items()}
        logger.info("Done!")
    
    def save(self, filename="vocab.pkl"):
        save_as_pickle(filename, self)
        logger.info("Saved vocab!")
    
    def add_ner(self, ner):
        self.ner2idx[ner] = len(self.ner2idx)
        self.idx2ner[len(self.ner2idx)] = ner
        logger.info("Added %s" % ner)
        self.save()
    
    def add_word(self, word):
        self.word2idx[word] = len(self.word2idx)
        self.idx2word[len(self.word2idx)] = word
        logger.info("Added %s" % word)
        self.save() 