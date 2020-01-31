# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:55:15 2019

@author: tsd
"""
import torch
from .tokenization_albert import AlbertTokenizer
from .preprocessing_funcs import save_as_pickle, load_pickle
from .ALBERT import AlbertForSequenceClassification
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class XLNet_infer(object):
    def __init__(self,):
        super(XLNet_infer, self).__init__()
        logger.info("Loading fine-tuned XLNet...")
        self.args = load_pickle("./data/args.pkl")
        self.net = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=self.args.num_classes)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=False)
        logger.info("Done!")
        
    def classify(self, text=None):
        if text is not None:
            logger.info("Classifying text...")
            text = self.tokenizer.tokenize(text)
            text = self.tokenizer.convert_tokens_to_ids(text[:(self.args.tokens_length-1)]).long().unsqueeze(0)
            self.net.eval()
            with torch.no_grad():
                outputs, _ = self.net(text)
                _, predicted = torch.max(outputs.data, 1)
            print("Predicted label: %d" % predicted.item())
        else:
            while True:
                text = input("Input text to classify: \n")
                if text in ["quit", "exit"]:
                    break
                logger.info("Classifying text...")
                text = self.tokenizer.tokenize(text)
                text = self.tokenizer.convert_tokens_to_ids(text[:(self.args.tokens_length-1)]).long().unsqueeze(0)
                self.net.eval()
                with torch.no_grad():
                    outputs, _ = self.net(text)
                    _, predicted = torch.max(outputs.data, 1)
                print("Predicted label: %d" % predicted.item()) 
