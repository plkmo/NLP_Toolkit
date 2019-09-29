# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:04:38 2019

@author: WT
"""

from utils.misc import save_as_pickle
from classification.models.GCN.trainer import train_and_fit as GCN
from classification.models.BERT.trainer import train_and_fit as BERT
from classification.models.XLNet.trainer import train_and_fit as XLNet
from classification.models.infer import infer_from_trained
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')


'''
Classification input data files:
train.csv - columns: text, labels
infer.csv - columns: text, optional labels

Output file:
results.csv - columns: index, predicted labels
'''

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default="./data/train.csv", help="training data csv file path")
    parser.add_argument("--infer_data", type=str, default="./data/infer.csv", help="infer data csv file path")
    parser.add_argument("--max_vocab_len", type=int, default=7000, help="GCN: Max vocab size to consider based on top frequency tokens")
    parser.add_argument("--hidden_size_1", type=int, default=330, help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=130, help="Size of second GCN hidden weights")
    parser.add_argument("--tokens_length", type=int, default=200, help="Max tokens length for BERT")
    parser.add_argument("--num_classes", type=int, default=66, help="Number of prediction classes (starts from integer 0)")
    parser.add_argument("--train_test_split", type=int, default=0, help="0: No, 1: Yes (Only activate if infer.csv contains labelled data)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="GCN: Ratio of test to training nodes")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--num_epochs", type=int, default=1700, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.0031, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID: (0: Graph Convolution Network (GCN), 1: BERT, 2: XLNet)")
    
    parser.add_argument("--train", type=int, default=1, help="Train model on dataset")
    parser.add_argument("--infer", type=int, default=0, help="Infer input sentence labels from trained model")
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    if args.train:
        if args.model_no == 0:
            GCN(args)
        elif args.model_no == 1:
            BERT(args)
        elif args.model_no == 2:
            XLNet(args)
    
    if args.infer:
        if args.model_no == 0:
            logger.info("Infer function not compatible with GCN!")
        else:
            inferer = infer_from_trained(args)
            while True:
                opt = input("Choose an option:\n0: Infer from stdin user input\n1: Infer from file (Input file: \'.\data\input.txt\'\
                                                                                                 Output file: '.\data\output.txt\'\n")
                if opt == '0':
                    inferer.infer_from_input()
                    break
                elif opt == '1':
                    inferer.infer_from_file()
                    break
                else:
                    print("Invalid option, please try again.")
