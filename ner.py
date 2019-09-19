# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 18:04:45 2019

@author: WT
"""
from ner.preprocessing_funcs import load_dataloaders
from ner.train_funcs import load_model_and_optimizer
from ner.trainer import train_and_fit
from utils.misc import save_as_pickle
from argparse import ArgumentParser
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./data/ner/conll2003/eng.train.txt", help="Path to training data txt file")
    parser.add_argument("--test_path", type=str, default="./data/ner/conll2003/eng.testa.txt", help="Path to test data txt file (if any)")
    parser.add_argument("--num_classes", type=int, default=9, help="Number of prediction classes (starts from integer 0)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--tokens_length", type=int, default=75, help="Max tokens length for BERT")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--num_epochs", type=int, default=125, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID: (0: BERT, 1: XLNet)")
    
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    #train_loader, a, test_loader, b = load_dataloaders(args)
    #net = load_model_and_optimizer(args)
    train_and_fit(args)