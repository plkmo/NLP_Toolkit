# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:35:16 2019

@author: WT
"""
import spacy
import re
from string import ascii_lowercase
from tqdm import tqdm

class tokener(object):
    def __init__(self, lang="en"):
        d = {"en":"en_core_web_lg", "fr":"fr_core_news_sm"}
        self.ob = spacy.load(d[lang])
    
    def tokenize(self, sent):
        sent = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sent))
        sent = re.sub(r"\!+", "!", sent)
        sent = re.sub(r"\,+", ",", sent)
        sent = re.sub(r"\?+", "?", sent)
        sent = re.sub(r"[ ]+", " ", sent)
        sent = sent.lower()
        sent = [token.text for token in self.ob.tokenizer(sent) if token.text != " "]
        return sent

class vocab(object):
    def __init__(self, level="word", model="transformer"):
        self.model = model
        if model == "transformer":
            self.w2idx = {"<sos>":0, "<eos>":2, "<pad>":1}
            self.idx2w = {0:"<sos>", 2:"<eos>", 1:"<pad>"}
            self.idx = 3
            self.level = level
                
        elif model == "h_encoder_decoder":
            self.w2idx = {"<sos>":0, "<eos>":2, "<pad>":1, "<sod>":3, "<eod>":4}
            self.idx2w = {0:"<sos>", 2:"<eos>", 1:"<pad>", 3:"<sod>", 4:"<eod>"}
            self.idx = 5
            self.level = level
        
    def build_vocab(self, df_text):
        if self.level == "word":
            word_soup = set([word for text in df_text for word in text])
            print("Building word vocab...")
            for word in tqdm(word_soup):
                if word not in self.w2idx.keys():
                    self.w2idx[word] = self.idx
                    self.idx += 1
                    
        elif self.level == "char":
            self.w2idx.update({k:v for k,v in zip(ascii_lowercase, [i for i in range(self.idx, len(ascii_lowercase) + self.idx)])})
            self.idx += len(ascii_lowercase)
            self.w2idx[" "] = self.idx; self.idx += 1
            self.w2idx["'"] = self.idx; self.idx += 1
            
        self.idx2w.update({v:k for k,v in self.w2idx.items() if v not in self.idx2w.keys()})
        
    def convert_w2idx(self, word_list):
        if self.level == "word":
            w = []
            for word in word_list:
                w.extend([self.w2idx[word]])
            return w
        
        elif self.level == "char":
            return [self.w2idx[c] for c in " ".join(word_list)]
    
    def convert_idx2w(self, idx_list):
        if self.level == "word":
            return [self.idx2w[idx] for idx in idx_list]
        elif self.level == "char":
            return [self.idx2w[idx] for idx in idx_list]