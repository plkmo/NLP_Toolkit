# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:09:24 2019

@author: WT
"""
from nlptoolkit.generation.infer import infer_from_trained
from nlptoolkit.utils.misc import save_as_pickle
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_no", type=int, default=2, help="0: GPT-2 ; 1: CTRL, 2: DialoGPT")
    args = parser.parse_args()
    
    save_as_pickle("args.pkl", args)
    
    inferer = infer_from_trained(args, tokens_len=70, top_k_beam=3)
    inferer.infer_from_input()