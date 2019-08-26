# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 07:59:03 2019

@author: tsd
"""
import torch
from pytorch_transformers import  GPT2Tokenizer
from .models.GPT2 import GPT2LMHeadModel

def infer_from_pretrained(input_sent="Hello, my dog is cute", tokens_len=100):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    if input_sent == None:
        while True:
            input_sent = input("Type your input sentence: \n")
            if input_sent.lower() in ['quit', 'exit']:
                break
            out_idxs = []
            input_ids = torch.tensor(tokenizer.encode(input_sent)).unsqueeze(0)  # Batch size 1
            for _ in range(tokens_len):
                outputs = model(input_ids, labels=input_ids)
                loss, logits = outputs[:2]
                s = torch.softmax(logits, dim=-1).max(dim=-1)[1]
                input_ids = torch.cat((input_ids, s[:,-1:]), dim=-1)
                ss = list(s.squeeze().detach().numpy())
                out_idxs.append(ss[-1])
            print(tokenizer.decode(out_idxs))
    else:
        out_idxs = []
        input_ids = torch.tensor(tokenizer.encode(input_sent)).unsqueeze(0)  # Batch size 1
        for _ in range(tokens_len):
            outputs = model(input_ids, labels=input_ids)
            loss, logits = outputs[:2]
            s = torch.softmax(logits, dim=-1).max(dim=-1)[1]
            input_ids = torch.cat((input_ids, s[:,-1:]), dim=-1)
            ss = list(s.squeeze().detach().numpy())
            out_idxs.append(ss[-1])
        print(tokenizer.decode(out_idxs))
    return model, logits