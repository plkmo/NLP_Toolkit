# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:31:51 2019

@author: WT
"""
from translation.trainer import train_and_fit
from utils.misc import save_as_pickle
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer model dimension")
    parser.add_argument("--num", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=0.00003, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    parser.add_argument("--num_epochs", type=int, default=500, help="No of epochs")
    args = parser.parse_args()
    
    save_as_pickle("args.pkl", args)
    
    train_and_fit(args)
    