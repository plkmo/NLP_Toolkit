# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:26:35 2019

@author: tsd
"""

import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class trg2_vocab_obj(object):
    def __init__(self, idx_mappings, mappings):
        map2 = {}
        for punc in mappings.keys():
            map2[punc] = idx_mappings[mappings[punc]]
        map2['word'] = len(map2)
        map2['sos'] = len(map2)
        map2['eos'] = len(map2)
        map2['pad'] = len(map2)
        self.punc2idx = map2
        self.idx2punc = {v:k for k,v in map2.items()}

def infer(args, from_data=False):
    return