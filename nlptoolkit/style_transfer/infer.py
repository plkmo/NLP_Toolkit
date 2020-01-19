#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 09:04:40 2020

@author: tsd
"""

import torch
from .data import load_dataset
from .models import StyleTransformer, Discriminator
from .add_misc.misc import save_as_pickle, load_pickle, Config
import spacy
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class infer_from_trained(object):
    def __init__(self, F_path='./save/Jan18190541/ckpts/900_F.pth', \
                 D_path = './save/Jan18190541/ckpts/900_D.pth',\
                 config_file='config.pkl', generator_only=True):
        
        logger.info("Loading vocab, model from trained...")
        self.cuda = torch.cuda.is_available()
        self.config = load_pickle(config_file)
        _, _, _, self.vocab = load_dataset(self.config)
        print('Vocab size:', len(self.vocab))
        
        self.model_F = StyleTransformer(self.config, self.vocab).to(self.config.device)
        self.model_F.load_state_dict(torch.load(F_path))
        self.model_F.eval()
        
        if not generator_only:
            self.model_D = Discriminator(self.config, self.vocab).to(self.config.device)
            self.model_D.load_state_dict(torch.load(D_path))
            self.model_D.eval()
            
        self.lang = spacy.load("en_core_web_lg")
        self.temperature = 1
        logger.info("Done!")
        
    def infer_sentence(self, sent='Super long wait, but worth it.', target_style=0):
        sent = [token.text.lower() for token in self.lang.tokenizer(sent) if token.text != ' ' ] + ['<eos>']
        sent_idx = [self.vocab.stoi[w] for w in sent]
        sent_tensor = torch.tensor(sent_idx).unsqueeze(0)
        rev_styles = torch.tensor([target_style])
        
        if self.cuda:
            sent_tensor, rev_styles = sent_tensor.cuda(), rev_styles.cuda()
        
        gen_log_probs = self.model_F(
            sent_tensor,
            None,
            torch.tensor(sent_tensor.shape[1]).cuda() if self.cuda else torch.tensor(sent_tensor.shape[1]),
            rev_styles,
            generate=True,
            differentiable_decode=True,
            temperature=self.temperature,
        )
        
        gen_idxs = torch.softmax(gen_log_probs, dim=1).max(-1)[1].squeeze()
        gen_sent = []
        for i in gen_idxs:
            token = self.vocab.itos[i.item()]
            if token == '<eos>':
                break
            else:
                gen_sent.append(token)
        gen_sent = " ".join(gen_sent)
        return gen_sent

if __name__ == '__main__':

    inferer = infer_from_trained(F_path='./data/style_transfer/Jan18190541/ckpts/900_F.pth', \
                                 D_path = './data/style_transfer/Jan18190541/ckpts/900_D.pth',\
                                 config_file='config.pkl', generator_only=True)
    
    gen_sent = inferer.infer_sentence(sent='Super long wait, but worth it.', target_style=0)
    print(gen_sent)