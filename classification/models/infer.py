# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:33:49 2019

@author: plkmo
"""
import pandas as pd
import torch
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class infer_from_trained(object):
    def __init__(self, args):
        self.args = args
        self.cuda = torch.cuda.is_available()
        logger.info("Loading tokenizer and model...")
        if args.model_no == 1:
            from .BERT.tokenization_bert import BertTokenizer as model_tokenizer
            from .BERT.BERT import BertForSequenceClassification as net
            from .BERT.train_funcs import load_state
            model_type = 'bert-base-uncased'
            
        elif args.model_no == 2:
            from .XLNet.tokenization_xlnet import XLNetTokenizer as model_tokenizer
            from .XLNet.XLNet import XLNetForSequenceClassification as net
            from .XLNet.train_funcs import load_state
            model_type = 'xlnet-base-cased'
            
        self.tokenizer = model_tokenizer.from_pretrained(model_type)
        self.tokens_length = args.tokens_length # max tokens length
        
        self.net = net.from_pretrained(model_type, num_labels=args.num_classes)
        if self.cuda:
            self.net.cuda()
        _, _ = load_state(self.net, None, None, args, load_best=False) 
        logger.info("Done!")
    
    def infer_sentence(self, sentence):
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
        else:
            outputs, _ = self.net(sentence, token_type_ids=type_ids, attention_mask=src_mask)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().item() if self.cuda else predicted.item()
        print("Predicted class: %d" % predicted)
        return predicted
    
    def infer_from_input(self):
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
            