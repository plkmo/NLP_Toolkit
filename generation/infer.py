# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 07:59:03 2019

@author: tsd
"""
import torch
from pytorch_transformers import  GPT2Tokenizer
from .models.GPT2 import GPT2LMHeadModel

def infer_from_pretrained(input_sent="Hello, my dog is cute"):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    if input_sent == None:
        while True:
            input_sent = input("Type your input sentence: \n")
            if input_sent.lower() in ['quit', 'exit']:
                break
            input_ids = torch.tensor(tokenizer.encode(input_sent)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
    else:
        input_ids = torch.tensor(tokenizer.encode(input_sent)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
    return logits