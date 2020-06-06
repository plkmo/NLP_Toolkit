# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:05:17 2019

@author: tsd
"""
from nlptoolkit.punctuation_restoration.trainer import train_and_fit
from nlptoolkit.punctuation_restoration.infer import infer_from_trained
from nlptoolkit.utils.misc import save_as_pickle
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/train.tags.en-fr.en", help="path to text file")
    parser.add_argument("--level", type=str, default="bpe", help="Level of tokenization (word, char or bpe)")
    parser.add_argument("--bpe_word_ratio", type=float, default=0.7, help="Ratio of BPE to word vocab")
    parser.add_argument("--bpe_vocab_size", type=int, default=7000, help="Size of bpe vocab if bpe is used")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer model dimension")
    parser.add_argument("--ff_dim", type=int, default=2048, help="Transformer feed-forward layer dimension")
    parser.add_argument("--num", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max_encoder_len", type=int, default=80, help="Max src length")
    parser.add_argument("--max_decoder_len", type=int, default=80, help="Max trg length")
    parser.add_argument("--LAS_embed_dim", type=int, default=512, help="PuncLSTM Embedding dimension")
    parser.add_argument("--LAS_hidden_size", type=int, default=512, help="PuncLSTM listener hidden_size")
    parser.add_argument("--num_epochs", type=int, default=127, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00003, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--T_max", type=int, default=5000, help="number of iterations before LR restart")
    parser.add_argument("--model_no", type=int, default=1, help="Model ID - 0: PuncTransformer\n\
                        1: PuncLSTM\n\
                        2: pyTransformer")
    
    parser.add_argument("--train", type=int, default=0, help="Train model on dataset")
    parser.add_argument("--infer", type=int, default=1, help="Infer input sentence labels from trained model")
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    if args.train:
        train_and_fit(args)
    if args.infer:
        inferer = infer_from_trained()
        inferer.infer_from_data()
