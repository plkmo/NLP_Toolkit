# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 07:59:03 2019

@author: tsd
"""
import torch
import numpy as np
from pytorch_transformers import  GPT2Tokenizer
from .models.GPT2 import GPT2LMHeadModel
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def top_k(logits, top_k_beam):
    s = torch.softmax(logits, dim=-1)[:,-1,:].topk(top_k_beam, dim=-1)[1]
    if logits.is_cuda:
        idx = np.random.choice(s.cpu().numpy()[0])
        idx = torch.tensor([[1, idx]]).cuda()
    else:
        idx = np.random.choice(s.numpy()[0])
        idx = torch.tensor([[1, idx]])
    return idx

def infer_from_pretrained(input_sent="Hello, my dog is cute", tokens_len=100, top_k_beam=None):
    logger.info("Loading pre-trained...")
    cuda = torch.cuda.is_available()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    logger.info("Loaded!")
    if cuda:
        model.cuda()
        
    if input_sent == None:
        while True:
            input_sent = input("Type your input sentence: \n")
            past = None
            if input_sent.lower() in ['quit', 'exit']:
                break
            out_idxs = []
            input_ids = torch.tensor(tokenizer.encode(input_sent)).unsqueeze(0)  # Batch size 1
            if cuda:
                input_ids = input_ids.cuda()
            
            logger.info("Generating...")
            for _ in tqdm(range(tokens_len)):
                if past is not None:
                    outputs = model(input_ids, labels=input_ids, past=past)
                else:
                    outputs = model(input_ids, labels=input_ids)
                loss, logits, _ = outputs
                if top_k_beam is not None:
                    s = top_k(logits, top_k_beam)
                else:
                    s = torch.softmax(logits, dim=-1).max(dim=-1)[1]
                input_ids = torch.cat((input_ids, s[:,-1:]), dim=-1)
                if cuda:
                    ss = list(s.squeeze().detach().cpu().numpy())
                else:
                    ss = list(s.squeeze().detach().numpy())
                out_idxs.append(ss[-1])
            print(tokenizer.decode(out_idxs))
            
    else:
        out_idxs = []
        input_ids = torch.tensor(tokenizer.encode(input_sent)).unsqueeze(0)  # Batch size 1
        if cuda:
            input_ids = input_ids.cuda()
        past = None
        
        logger.info("Generating...")
        for _ in tqdm(range(tokens_len)):
            if past is not None:
                outputs = model(input_ids, labels=input_ids, past=past)
            else:
                outputs = model(input_ids, labels=input_ids)
            loss, logits, _ = outputs
            if top_k_beam is not None:
                s = top_k(logits, top_k_beam)
            else:
                s = torch.softmax(logits, dim=-1).max(dim=-1)[1]
            input_ids = torch.cat((input_ids, s[:,-1:]), dim=-1)
            if cuda:
                ss = list(s.squeeze().detach().cpu().numpy())
            else:
                ss = list(s.squeeze().detach().numpy())
            out_idxs.append(ss[-1])
        print(tokenizer.decode(out_idxs))
    return outputs