# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:05:17 2019

@author: tsd
"""
from punctuation_restoration.preprocessing_funcs import load_dataloaders
from utils.misc import save_as_pickle
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    df = load_dataloaders(args)