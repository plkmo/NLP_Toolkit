# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:04:38 2019

@author: WT
"""

from utils.misc import save_as_pickle
from classification.models.GCN.trainer import train_and_fit as GCN
from classification.models.BERT.trainer import train_and_fit as BERT
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default="./data/train.csv", help="training data csv file path")
    parser.add_argument("--infer_data", type=str, default="./data/infer.csv", help="infer data csv file path")
    parser.add_argument("--max_vocab_len", type=int, default=7000, help="GCN: Max vocab size to consider based on top frequency tokens")
    parser.add_argument("--hidden_size_1", type=int, default=330, help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=130, help="Size of second GCN hidden weights")
    parser.add_argument("--num_classes", type=int, default=66, help="Number of prediction classes")
    parser.add_argument("--train_test_split", type=int, default=0, help="GCN: Split train data for testing. (0: No, 1: Yes)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="GCN: Ratio of test to training nodes")
    parser.add_argument("--num_epochs", type=int, default=1700, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.011, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID: (0: Graph Convolution Network (GCN), 1: BERT)")
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    if args.model_no == 0:
        GCN(args)
    elif args.model_no == 1:
        BERT(args)