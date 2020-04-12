# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:31:51 2019

@author: WT
"""
from nlptoolkit.translation.trainer import train_and_fit
from nlptoolkit.translation.infer import infer_from_trained, evaluate_corpus_bleu
from nlptoolkit.utils.misc import save_as_pickle
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, default="./data/translation/eng_zh/news-commentary-v13.zh-en.en", help="Path to source data txt file")
    parser.add_argument("--trg_path", type=str, default="./data/translation/eng_zh/news-commentary-v13.zh-en.zh", help="Path to target data txt file")
    parser.add_argument("--src_lang", type=str, default="en", help="src language: en (English), fr (French), zh (Chinese)")
    parser.add_argument("--trg_lang", type=str, default="zh", help="trg language: en (English), fr (French), zh (Chinese)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer model dimension")
    parser.add_argument("--ff_dim", type=int, default=2048, help="Transformer feed-forward layer dimension")
    parser.add_argument("--num", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--max_encoder_len", type=int, default=200, help="Max src length")
    parser.add_argument("--max_decoder_len", type=int, default=200, help="Max trg length")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32")
    parser.add_argument("--num_epochs", type=int, default=280, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=3, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--T_max", type=int, default=7000, help="number of iterations before LR restart")
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID (0: Transformer\n'
                                                                    1: LightConv''')
    
    parser.add_argument("--train", type=int, default=1, help="Train model on dataset")
    parser.add_argument("--evaluate", type=int, default=1, help="Evaluate the trained model on dataset")
    parser.add_argument("--infer", type=int, default=1, help="Infer input sentences")
    args = parser.parse_args()
    
    save_as_pickle("args.pkl", args)
    
    '''PyTorch's transformer module runs much slower'''
    if args.train:
        train_and_fit(args, pytransformer=False)
    if args.evaluate:
        evaluate_corpus_bleu(args)
    if args.infer:
        inferer = infer_from_trained(args)
        inferer.infer_from_input()