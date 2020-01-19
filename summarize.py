# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:56:35 2019

@author: WT
"""
from nlptoolkit.utils.misc import save_as_pickle
from nlptoolkit.summarization.trainer import train_and_fit
from nlptoolkit.summarization.infer import infer_from_trained
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path1", type=str, default="./data/summarize_data/datasets/cnn_stories/cnn/stories/",\
                        help="Full path to CNN dataset")
    parser.add_argument("--data_path2", type=str, default="./data/summarize_data/datasets/dailymail_stories/dailymail/stories/",\
                        help="Full path to dailymail dataset (leave as empty string if none)")
    parser.add_argument("--level", type=str, default="bpe", help="Level of tokenization (word, char or bpe)")
    parser.add_argument("--bpe_word_ratio", type=float, default=0.7, help="Ratio of BPE to word vocab")
    parser.add_argument("--bpe_vocab_size", type=int, default=9000, help="Size of bpe vocab if bpe is used")
    parser.add_argument("--max_features_length", type=int, default=1000, help="Max length of features (word, char or bpe level)")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer model dimension")
    parser.add_argument("--ff_dim", type=int, default=256, help="Transformer Feed forward layer dimension")
    parser.add_argument("--num", type=int, default=6, help="Transformer number of layers per block")
    parser.add_argument("--n_heads", type=int, default=4, help="Transformer number of attention heads")
    parser.add_argument("--LAS_embed_dim", type=int, default=512, help="LAS Embedding dimension")
    parser.add_argument("--LAS_hidden_size", type=int, default=512, help="LAS listener hidden_size")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32")
    parser.add_argument("--train_test_ratio", type=float, default=0.9, help='Ratio for train-test split')
    parser.add_argument("--num_epochs", type=int, default=8000, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=5, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--T_max", type=int, default=7000, help="number of iterations before LR restart")
    parser.add_argument("--model_no", type=int, default=1, help="Model ID: 0 = Transformer, 1 = LAS")
    
    parser.add_argument("--train", type=int, default=1, help="Train model on dataset")
    parser.add_argument("--infer", type=int, default=1, help="Infer input sentence labels from trained model")
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    if args.train:
        train_and_fit(args)
    if args.infer:
        inferer = infer_from_trained(args)
        inferer.infer_from_data()