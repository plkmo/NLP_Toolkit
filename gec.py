# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:09:24 2019

@author: WT
"""
import os
from nlptoolkit.gec.trainer import train_and_fit
from nlptoolkit.gec.infer import infer_from_trained
from nlptoolkit.gec.models.gector.utils.preprocess_data import convert_data_from_raw_files
from nlptoolkit.utils.misc import save_as_pickle, split_dataset
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_no", type=int, default=0, help="0: GECToR")
    parser.add_argument('--model_path', type=str, default=['./data/gec/gector/roberta_1_gector.th'],
                        help='Path to the trained model file, if any', nargs='+')
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    
    ### GECToR
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'],
                        help='(For GECToR) Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='(For GECToR) Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    
    ### GECToR Training
    parser.add_argument('--model_dir', type=str,
                        default='./data/gec/gector/model_checkpoints/',
                        help='Path to the model dir')
    parser.add_argument('--src',
                        help='Path to the source data', type=str,
                        default='./data/gec/gector/train_data/a1_train_incorr_sentences.txt')
    parser.add_argument('--tgt',
                        help='Path to the target data', type=str,
                        default='./data/gec/gector/train_data/a1_train_corr_sentences.txt')
    parser.add_argument('--train_test_ratio', type=str, default=0.9,
                        help='Train test ratio')
    parser.add_argument('--train_set', type=str,
                        default='./data/gec/gector/train_data/a1_train.txt',
                        help='Path to the saved processed train data')
    parser.add_argument('--dev_set', type=str,
                        default='./data/gec/gector/train_data/a1_test.txt',
                        help='Path to the saved processed dev data')
    parser.add_argument('--target_vocab_size',
                        type=int,
                        help='The size of target vocabularies.',
                        default=1000)
    parser.add_argument('--n_epoch',
                        type=int,
                        help='The number of epoch for training model.',
                        default=20)
    parser.add_argument('--patience',
                        type=int,
                        help='The number of epoch with any improvements'
                             ' on validation set.',
                        default=3)
    parser.add_argument('--skip_correct',
                        type=int,
                        help='If set than correct sentences will be skipped '
                             'by data reader.',
                        default=1)
    parser.add_argument('--skip_complex',
                        type=int,
                        help='If set than complex corrections will be skipped '
                             'by data reader.',
                        choices=[0, 1, 2, 3, 4, 5],
                        default=0)
    parser.add_argument('--tune_bert',
                        type=int,
                        help='If more then 0 then fine tune bert.',
                        default=1)
    parser.add_argument('--tag_strategy',
                        choices=['keep_one', 'merge_all'],
                        help='The type of the data reader behaviour.',
                        default='keep_one')
    parser.add_argument('--accumulation_size',
                        type=int,
                        help='How many batches do you want accumulate.',
                        default=4)
    parser.add_argument('--lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=1e-5)
    parser.add_argument('--cold_steps_count',
                        type=int,
                        help='Whether to train only classifier layers first.',
                        default=4)
    parser.add_argument('--cold_lr',
                        type=float,
                        help='Learning rate during cold_steps.',
                        default=1e-3)
    parser.add_argument('--predictor_dropout',
                        type=float,
                        help='The value of dropout for predictor.',
                        default=0.0)
    parser.add_argument('--pieces_per_token',
                        type=int,
                        help='The max number for pieces per token.',
                        default=5)
    parser.add_argument('--cuda_verbose_steps',
                        help='Number of steps after which CUDA memory information is printed. '
                             'Makes sense for local testing. Usually about 1000.',
                        default=None)
    parser.add_argument('--label_smoothing',
                        type=float,
                        help='The value of parameter alpha for label smoothing.',
                        default=0.0)
    parser.add_argument('--tn_prob',
                        type=float,
                        help='The probability to take TN from data.',
                        default=0)
    parser.add_argument('--tp_prob',
                        type=float,
                        help='The probability to take TP from data.',
                        default=1)
    parser.add_argument('--updates_per_epoch',
                        type=int,
                        help='If set then each epoch will contain the exact amount of updates.',
                        default=0)
    parser.add_argument('--pretrain_folder',
                        help='The name of the pretrain folder.')
    parser.add_argument('--pretrain',
                        help='The name of the pretrain weights in pretrain_folder param.',
                        default='')
    
    ### GECToR inference
    parser.add_argument('--vocab_path', type=str, default='./data/gec/gector/output_vocabulary/',
                        help='(For GECToR) Path to the model file.')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='(For GECToR) How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--min_probability',
                        type=float,
                        help='(For GECToR inference)',
                        default=0.0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='(For GECToR inference)',
                        default=0.0)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='(For GECToR inference) Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='(For GECToR inference) Used to calculate weighted average', nargs='+',
                        default=None)
    
    
    parser.add_argument("--train", type=int, default=0, 
                        help="Train model on dataset")
    parser.add_argument("--infer", type=int, default=1, 
                        help="Infer input sentence from trained model")
    args = parser.parse_args()
    
    save_as_pickle("args.pkl", args)
    
    if args.train == 1:
        try: 
            # Example: Preprocess Dataset from synthetic (https://drive.google.com/file/d/1bl5reJ-XhPEfEaPjvO45M7w0yN-0XGOA/view) (a1 only)
            if not os.path.isfile('./data/gec/gector/train_data/a1_processed.txt'):
                convert_data_from_raw_files(source_file=args.src, target_file=args.tgt,
                                            output_file='./data/gec/gector/train_data/a1_processed.txt', 
                                            chunk_size=1000000)
            split_dataset(file='./data/gec/gector/train_data/a1_processed.txt', 
                          ratio=args.train_test_ratio, 
                          train=args.train_set, 
                          test=args.dev_set)
        except:
            pass
        train_and_fit(args)
    
    if args.infer == 1:
        inferer = infer_from_trained(args)
        inferer.infer_from_file(input_file='./data/gec/gector/input.txt', \
                                output_file='./data/gec/gector/output.txt', batch_size=32)
        print(inferer.infer_sentence('He has dog'))
        inferer.infer_from_input()