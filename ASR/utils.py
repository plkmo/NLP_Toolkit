# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:21:57 2019

@author: WT
"""
import os
import pickle
import spacy
import re
from string import ascii_lowercase
import torch
from torch.autograd import Variable
import numpy as np
import math

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
class tokener(object):
    def __init__(self, lang="en"):
        d = {"en":"en_core_web_sm", "fr":"fr_core_news_sm"}
        self.ob = spacy.load(d[lang])
    
    def tokenize(self, sent):
        sent = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sent))
        sent = re.sub(r"\!+", "!", sent)
        sent = re.sub(r"\,+", ",", sent)
        sent = re.sub(r"\?+", "?", sent)
        sent = re.sub(r"[ ]+", " ", sent)
        sent = sent.lower()
        sent = [token.text for token in self.ob.tokenizer(sent) if token.text != " "]
        return sent

class vocab(object):
    def __init__(self, level="word"):
        self.w2idx = {"<sos>":0, "<eos>":2, "<pad>":1}
        self.idx2w = {0:"<sos>", 2:"<eos>", 1:"<pad>"}
        self.idx = 3
        self.level = level
        
    def build_vocab(self, df_text):
        if self.level == "word":
            self.w2idx[" "] = self.idx; self.idx += 1 # space is index 3
            word_soup = set([word for text in df_text for word in text])
            for word in word_soup:
                if word not in self.w2idx.keys():
                    self.w2idx[word] = self.idx
                    self.idx += 1
        elif self.level == "char":
            self.w2idx.update({k:v for k,v in zip(ascii_lowercase, [i for i in range(self.idx, len(ascii_lowercase) + self.idx)])})
            self.idx += len(ascii_lowercase)
            self.w2idx[" "] = self.idx; self.idx += 1
            self.w2idx["'"] = self.idx; self.idx += 1
        self.idx2w.update({v:k for k,v in self.w2idx.items() if v not in self.idx2w.keys()})
        
    def convert_w2idx(self, word_list):
        if self.level == "word":
            w = []
            for word in word_list:
                w.extend([self.w2idx[word], 3])
            return w
        elif self.level == "char":
            return [self.w2idx[c] for c in " ".join(word_list)]
    
    def convert_idx2w(self, idx_list):
        if self.level == "word":
            return [self.idx2w[idx] for idx in idx_list]
        elif self.level == "char":
            return [self.idx2w[idx] for idx in idx_list]
        
class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    T_max : int
        The maximum number of iterations within the first cycle.
    eta_min : float, optional (default: 0)
        The minimum learning rate.
    last_epoch : int, optional (default: -1)
        The index of the last epoch.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs

def lrate(n, d_model, k=10, warmup_n=25000):
    lr = (k/math.sqrt(d_model))*min(1/math.sqrt(n), n*warmup_n**(-1.5))
    return lr

def CreateOnehotVariable(input_x, encoding_dim=64):
# This is a function to generate an one-hot encoded tensor with given batch size and index
# Input : input_x which is a Tensor or Variable with shape [batch size, timesteps]
#         encoding_dim, the number of classes of input
# Output: onehot_x, a Variable containing onehot vector with shape [batch size, timesteps, encoding_dim]
    if type(input_x) is Variable:
        input_x = input_x.data 
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1,input_x,1)).type(input_type)
    
    return onehot_x