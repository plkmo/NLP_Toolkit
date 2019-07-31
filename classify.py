# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:04:38 2019

@author: WT
"""

from classification.models.GCN.preprocessing_funcs import save_as_pickle
from classification.models.GCN.trainer import train
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default="./data/train.csv", help="training data csv file path")
    parser.add_argument("--infer_data", type=str, default="./data/infer.csv", help="infer data csv file path")
    parser.add_argument("--hidden_size_1", type=int, default=330, help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=130, help="Size of second GCN hidden weights")
    parser.add_argument("--num_classes", type=int, default=66, help="Number of prediction classes")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test to training nodes")
    parser.add_argument("--num_epochs", type=int, default=5000, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.011, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    train(args)
    
    logger.info("Evaluate results...")
    #evaluate_model_results(args=args)