# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 07:59:03 2019

@author: tsd
"""
import os
import pickle
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def top_k(logits, top_k_beam):
    s = torch.softmax(logits, dim=-1)[:,-1,:].topk(top_k_beam, dim=-1)[1]
    if logits.is_cuda:
        idx = np.random.choice(s.cpu().numpy()[0])
        idx = torch.tensor([[1, idx]]).cuda()
    else:
        idx = np.random.choice(s.numpy()[0])
        idx = torch.tensor([[1, idx]])
    return idx

class infer_from_trained(object):
    def __init__(self, args=None, tokens_len=100, top_k_beam=3):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()
        
        self.tokens_len = tokens_len
        self.top_k_beam = top_k_beam
        
        logger.info("Loading tokenizer and model...")
        if args.model_no == 0:
            from .models.GPT2.tokenization_gpt2 import GPT2Tokenizer
            from .models.GPT2.GPT2 import GPT2LMHeadModel
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        elif args.model_no == 1:
            from .models.CTRL.tokenization_ctrl import CTRLTokenizer
            from .models.CTRL.modeling_ctrl import CTRLLMHeadModel
            self.tokenizer = CTRLTokenizer.from_pretrained('ctrl')
            self.model = CTRLLMHeadModel.from_pretrained('ctrl')
        elif args.model_no == 2:
            from .models.DialoGPT.modeling_auto import AutoModelWithLMHead
            from .models.DialoGPT.tokenization_auto import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
        
        if self.cuda:
            self.model.cuda()
        
    def infer_sentence(self, sent):
        past = None
        out_idxs = []
        input_ids = torch.tensor(self.tokenizer.encode(sent)).unsqueeze(0)  # Batch size 1
        if self.cuda:
            input_ids = input_ids.cuda()
        
        logger.info("Generating...")
        for _ in tqdm(range(self.tokens_len)):
            if past is not None:
                outputs = self.model(input_ids, labels=input_ids, past=past)
            else:
                outputs = self.model(input_ids, labels=input_ids)
            loss, logits, _ = outputs
            if self.top_k_beam is not None:
                s = top_k(logits, self.top_k_beam)
            else:
                s = torch.softmax(logits, dim=-1).max(dim=-1)[1]
            input_ids = torch.cat((input_ids, s[:,-1:]), dim=-1)
            if self.cuda:
                ss = list(s.squeeze().detach().cpu().numpy())
            else:
                ss = list(s.squeeze().detach().numpy())
            out_idxs.append(ss[-1])
        out_sent = self.tokenizer.decode(out_idxs)
        print(out_sent)
        return out_sent
    
    def infer_from_input(self,):
        if self.args.model_no == 2:
            # Let's chat for 5 lines
            for step in range(5):
                # encode the new user input, add the eos_token and return a tensor in Pytorch
                new_user_input_ids = self.tokenizer.encode(input(">> User:") + self.tokenizer.eos_token,\
                                                           return_tensors='pt')
                
                if self.cuda:
                    new_user_input_ids = new_user_input_ids.cuda()
                # append the new user input tokens to the chat history
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], \
                                          dim=-1) if step > 0 else new_user_input_ids
            
                # generated a response while limiting the total chat history to 1000 tokens, 
                chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, \
                                                       pad_token_id=self.tokenizer.eos_token_id)
            
                # pretty print last ouput tokens from bot
                print("DialoGPT: {}".format(self.tokenizer.decode(chat_history_ids[:, \
                      bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        
        else:
            while True:
                with torch.no_grad():
                    input_sent = input("Type your input sentence: \n")
                    if input_sent.lower() in ['quit', 'exit']:
                        break
                    self.infer_sentence(input_sent)
        return
    
    
    def infer_from_file(self, in_file="./data/input.txt", out_file="./data/output.txt"):
        df = pd.read_csv(in_file, header=None, names=["sents"])
        df['generated'] = df.progress_apply(lambda x: self.infer_sentence(x['sents']), axis=1)
        df.to_csv(out_file, index=False)
        logger.info("Done and saved as %s!" % out_file)
        return

def infer_from_pretrained(input_sent="Hello, my dog is cute", tokens_len=100, top_k_beam=None):
    logger.info("Loading pre-trained...")
    cuda = torch.cuda.is_available()
    
    from .models.GPT2.tokenization_gpt2 import GPT2Tokenizer
    from .models.GPT2.GPT2 import GPT2LMHeadModel
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