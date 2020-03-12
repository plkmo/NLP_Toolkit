# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:33:49 2019

@author: plkmo
"""
import pickle
import os
import pandas as pd
import torch
from .DGI import DGI
from .train_funcs import load_datasets, get_X_A_hat, load_state
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class infer_from_trained(object):
    def __init__(self, args=None):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()
        
        logger.info("Loading tokenizer and model...")    
        self.G = load_datasets(args)
        X, A_hat = get_X_A_hat(self.G, corrupt=False)
        #print(labels_selected, labels_not_selected)
        self.net = DGI(X.shape[1], args)
        
        _, _ = load_state(self.net, None, None, model_no=args.model_no, load_best=False)
        
        if self.cuda:
            self.net.cuda()
        
        self.net.eval()
        logger.info("Done!")
    
    def infer_sentence(self, sentence):
        self.net.eval()
        sentence = self.tokenizer.tokenize("[CLS] " + sentence)
        sentence = self.tokenizer.convert_tokens_to_ids(sentence[:(self.args.tokens_length-1)] + ["[SEP]"])
        sentence = torch.tensor(sentence).unsqueeze(0)
        type_ids = torch.zeros([sentence.shape[0], sentence.shape[1]], requires_grad=False).long()
        src_mask = (sentence != 0).long()
        if self.cuda:
            sentence = sentence.cuda()
            type_ids = type_ids.cuda()
            src_mask = src_mask.cuda()
        if self.args.model_no == 1:
            outputs = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
        elif self.args.model_no in [4, 5]:
            outputs = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
            outputs = outputs[0]
        else:
            outputs, _ = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().item() if self.cuda else predicted.item()
        print("Predicted class: %d" % predicted)
        return predicted
    
    def infer_from_input(self):
        self.net.eval()
        while True:
            user_input = input("Type input sentence (Type \'exit' or \'quit' to quit):\n")
            if user_input in ["exit", "quit"]:
                break
            predicted = self.infer_sentence(user_input)
        return predicted
    
    def infer_from_file(self, in_file="./data/input.txt", out_file="./data/output.txt"):
        df = pd.read_csv(in_file, header=None, names=["sents"])
        df['labels'] = df.progress_apply(lambda x: self.infer_sentence(x['sents']), axis=1)
        df.to_csv(out_file, index=False)
        logger.info("Done and saved as %s!" % out_file)
        return
            
