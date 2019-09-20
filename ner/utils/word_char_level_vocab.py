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
    def __init__(self, df_train=None, df_test=None, ner_only=True):
        
        if df_train is not None:
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
            
            self.word2idx = {k:v for v, k in enumerate(sents, 0)}
            self.word2idx['<pad>'] = -100
            self.ner2idx = {k:v for v, k in enumerate(ners, 0)}
            self.ner2idx.update({'<pad>':-100, 'B-PER':len(self.ner2idx)}) #, '<sos>':1, '<eos>':2})
            self.idx2word = {v:k for k,v in self.word2idx.items()}
            self.idx2ner = {v:k for k, v in self.ner2idx.items()}
            logger.info("Done!")
        else:
            '''
            self.ner2idx = {'I-ORG': 1,  
                            'I-MISC': 2,
                            'I-LOC': 3,
                            'I-PER': 4,
                            'B-MISC': 5,
                            'B-LOC': 6,
                            'B-ORG': 7,
                            'O': 8,
                            '<pad>': -9,
                            'B-PER': 0}
            '''
            self.ner2idx = {"O":0, "B-MISC":1, "I-MISC":2,  "B-PER":3, "I-PER":4, "B-ORG":5, "I-ORG":6, "B-LOC":7, "I-LOC":8,\
                            '<pad':-100}
            self.idx2ner = {v:k for k, v in self.ner2idx.items()}
    
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