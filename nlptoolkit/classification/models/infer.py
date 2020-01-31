# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:33:49 2019

@author: plkmo
"""
import pickle
import os
import pandas as pd
import torch
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
        if self.args.model_no == 1:
            from .BERT.tokenization_bert import BertTokenizer as model_tokenizer
            from .BERT.BERT import BertForSequenceClassification as net
            from .BERT.train_funcs import load_state
            model_type = 'bert-base-uncased'
            lower_case = True
            
        elif self.args.model_no == 2:
            from .XLNet.tokenization_xlnet import XLNetTokenizer as model_tokenizer
            from .XLNet.XLNet import XLNetForSequenceClassification as net
            from .XLNet.train_funcs import load_state
            model_type = 'xlnet-base-cased'
            lower_case = False
            
        elif self.args.model_no == 4:
            from .ALBERT.tokenization_albert import AlbertTokenizer as model_tokenizer
            from .ALBERT.ALBERT import AlbertForSequenceClassification as net
            from .ALBERT.train_funcs import load_state
            model_type = 'albert-base-v2'
            lower_case = False
            
        elif self.args.model_no == 5:
            from .XLMRoBERTa.tokenization_xlm_roberta import XLMRobertaTokenizer as model_tokenizer
            from .XLMRoBERTa.XLMRoBERTa import XLMRobertaForSequenceClassification as net
            from .XLMRoBERTa.train_funcs import load_state
            model_type = 'xlm-roberta-base'
            lower_case = False
            
        self.tokenizer = model_tokenizer.from_pretrained(model_type, do_lower_case=lower_case)
        self.tokens_length = args.tokens_length # max tokens length
        
        self.net = net.from_pretrained(model_type, num_labels=args.num_classes)
        if self.cuda:
            self.net.cuda()
        _, _ = load_state(self.net, None, None, args, load_best=False) 
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
            
