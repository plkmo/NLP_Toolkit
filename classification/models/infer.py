# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:33:49 2019

@author: plkmo
"""

import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class infer_from_trained(object):
    def __init__(self, args):
        if args.model_no == 1:
            from .BERT.tokenization_bert import BertTokenizer as model_tokenizer
            model_type = 'bert-base-uncased'
        elif args.model_no == 2:
            pass
        tokenizer = model_tokenizer.from_pretrained(model_type)
        tokens_length = args.tokens_length # max tokens length