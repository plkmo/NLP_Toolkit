# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:25:58 2019

@author: WT
"""

import os
import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import logging
from .utils.misc_utils import save_as_pickle, load_pickle
from .utils.word_char_level_vocab import vocab_mapper
from .models.BERT.tokenization_bert import BertTokenizer

tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def pad_sos_eos(x, sos, eos):
    return [sos] + x + [eos] 

class args():
    def __init__(self):
        self.batch_size = 5

def clean_and_tokenize_text(text, table, tokenizer, clean_only=False):
    if isinstance(text, str):
        text = text.replace("(CNN) -- ","").replace("U.N.", "UN").replace("U.S.", "USA")
        text = text.replace(".", ". ").replace(",", ", ").replace("?", "? ").replace("!", "! ")
        text = text.translate(table)
        if clean_only == False:
            text = tokenizer.tokenize(text)
            text = [w for w in text if not any(char.isdigit() for char in w)]
        return text
    
def get_POS_conll2003_data(args, load_extracted=True):
    """
    Extracts POS dataset, saves then
    returns dataframe containing body (main text) and POS tags columns
    table: table containing symbols to remove from text
    tokenizer: tokenizer to tokenize text into word tokens
    """
    train_path = args.train_path
    if args.test_path is not None:
        test_path = args.test_path
    else:
        test_path = None
        df_test = None
        
    if load_extracted:
        logger.info("Loading pre-processed saved files...")
        df_train =  load_pickle("df_train.pkl")
        if os.path.isfile("./data/df_test.pkl") is not None:
            df_test = load_pickle("df_test.pkl")
        else:
            df_test = None
        logger.info("Loaded!")
            
    else:
        logger.info("Extracting data...")
        with open(train_path, "r", encoding="utf8") as f:
            text = f.readlines()
        sents, ners = [], []
        sent, sent_ner = [], []
        for line in tqdm(text):
            line = line.split(" ")
            if len(line) == 4:
                word, pos, btag, ner = line
                if word != '-DOCSTART-':
                    sent.append(word.lower()); sent_ner.append(re.sub("\n", "", pos))
            else:
                sents.append(sent); ners.append(sent_ner)
                sent, sent_ner = [], []
        assert len(sents) == len(ners)
        df_train = pd.DataFrame(data={"sents":sents, "poss":ners})
        df_train['length'] = df_train.progress_apply(lambda x: len(x['poss']), axis=1)
        df_train = df_train[df_train['length'] != 0]
        ## to find num_classes
        num_pos = []
        for p in df_train['poss']:
            num_pos.extend(p)
        num_classes = len(list(set(num_pos)))
        logger.info("Number of unique POS tags found: %d" % num_classes)
        
        if test_path is not None:
            with open(test_path, "r", encoding="utf8") as f:
                text = f.readlines()
            sents, ners = [], []
            sent, sent_ner = [], []
            for line in tqdm(text):
                line = line.split(" ")
                if len(line) == 4:
                    word, pos, btag, ner = line
                    if word != '-DOCSTART-':
                        sent.append(word.lower()); sent_ner.append(re.sub("\n", "", pos))
                else:
                    sents.append(sent); ners.append(sent_ner)
                    sent, sent_ner = [], []
            assert len(sents) == len(ners)
            df_test = pd.DataFrame(data={"sents":sents, "poss":ners})
            df_test['length'] = df_test.progress_apply(lambda x: len(x['poss']), axis=1)
            df_test = df_test[df_test['length'] != 0]
                
    return df_train, df_test

def get_POS_twitter_data(args, load_extracted=True):
    """
    Extracts POS twitter dataset, saves then
    returns dataframe containing body (main text) and POS tags columns
    table: table containing symbols to remove from text
    tokenizer: tokenizer to tokenize text into word tokens
    """
    train_path = args.train_path
    if args.test_path != "":
        test_path = args.test_path
    else:
        test_path = None
        df_test = None
        
    if load_extracted:
        logger.info("Loading pre-processed saved files...")
        df_train =  load_pickle("df_train.pkl")
        if os.path.isfile("./data/df_test.pkl") is not None:
            df_test = load_pickle("df_test.pkl")
        else:
            df_test = None
        logger.info("Loaded!")
            
    else:
        logger.info("Extracting data...")
        with open(train_path, "r", encoding="utf8") as f:
            text = f.readlines()
            
        sents, poss = [], []
        sent, sent_pos = [], []
        for line in tqdm(text):
            line = line.split()
            if len(line) == 2:
                pos, word = line
                sent.append(word.lower()); sent_pos.append(re.sub("\n", "", pos))
            else:
                assert len(sent) == len(sent_pos)
                sents.append(sent); poss.append(sent_pos)
                sent, sent_pos = [], []
        assert len(sents) == len(poss)
        
        df_train = pd.DataFrame(data={"sents":sents, "poss":poss})
        df_train['length'] = df_train.progress_apply(lambda x: len(x['poss']), axis=1)
        df_train = df_train[df_train['length'] != 0]
        
        if test_path is not None:
            with open(test_path, "r", encoding="utf8") as f:
                text = f.readlines()
                
            sents, poss = [], []
            sent, sent_pos = [], []
            for line in tqdm(text):
                line = line.split()
                if len(line) == 2:
                    pos, word = line
                    sent.append(word.lower()); sent_pos.append(re.sub("\n", "", pos))
                else:
                    assert len(sent) == len(sent_pos)
                    sents.append(sent); poss.append(sent_pos)
                    sent, sent_pos = [], []
            assert len(sents) == len(poss)
            
            df_test = pd.DataFrame(data={"sents":sents, "poss":poss})
            df_test['length'] = df_test.progress_apply(lambda x: len(x['poss']), axis=1)
            df_test = df_test[df_test['length'] != 0]
            
    return df_train, df_test

def convert_poss_to_ids(poss, vocab):
    return [vocab.pos2idx[pos] for pos in poss]

def generate_pos_ids_labels(sent_tokens, sent_poss, tokenizer, vocab):
    pos_pad_id = vocab.pos2idx['<pad>']
    sent_ids, poss_ids = [], []
    for word, pos in zip(sent_tokens, sent_poss):
        sub_word_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        pos_ids = convert_poss_to_ids([pos], vocab) + [pos_pad_id]*(len(sub_word_ids) - 1)
        sent_ids.extend(sub_word_ids); poss_ids.extend(pos_ids)
    return sent_ids, poss_ids

def pos_preprocess(args, df_train, df_test=None, include_cls=True):
    logger.info("Preprocessing...")
    vocab = vocab_mapper(df_train, df_test)
    vocab.save()
    
    logger.info("Tokenizing...")
    if args.model_no == 0: # BERT
        max_len = args.tokens_length
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
        pos_pad_id = vocab.pos2idx['<pad>']
        cls_id = tokenizer.convert_tokens_to_ids(["[CLS]"])
        sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])
        
        if include_cls:
            df_train['sents_ids'] = df_train.progress_apply(lambda x: generate_pos_ids_labels(x['sents'], x['poss'], tokenizer, vocab),\
                                                                axis=1)
            df_train['poss_ids'] = df_train.progress_apply(lambda x: [pos_pad_id] + x['sents_ids'][1][:(max_len - 2)] + [pos_pad_id], axis=1)
            df_train['sents_ids'] = df_train.progress_apply(lambda x: cls_id + x['sents_ids'][0][:(max_len - 2)] + sep_id, axis=1)
            
        else:
            df_train['sents_ids'] = df_train.progress_apply(lambda x: generate_pos_ids_labels(x['sents'], x['poss'], tokenizer, vocab),\
                                                                axis=1)
            df_train['poss_ids'] = df_train.progress_apply(lambda x: x['sents_ids'][1][:(max_len - 2)], axis=1)
            df_train['sents_ids'] = df_train.progress_apply(lambda x: x['sents_ids'][0][:(max_len - 2)], axis=1)
        save_as_pickle("df_train.pkl", df_train)
        
        if df_test is not None:
            if include_cls:
                df_test['sents_ids'] = df_test.progress_apply(lambda x: generate_pos_ids_labels(x['sents'], x['poss'], tokenizer, vocab),\
                                                                    axis=1)
                df_test['poss_ids'] = df_test.progress_apply(lambda x: [pos_pad_id] + x['sents_ids'][1][:(max_len - 2)] + [pos_pad_id], axis=1)
                df_test['sents_ids'] = df_test.progress_apply(lambda x: cls_id + x['sents_ids'][0][:(max_len - 2)] + sep_id, axis=1)
                
            else:
                df_test['sents_ids'] = df_test.progress_apply(lambda x: generate_pos_ids_labels(x['sents'], x['poss'], tokenizer, vocab),\
                                                                    axis=1)
                df_test['poss_ids'] = df_test.progress_apply(lambda x: x['sents_ids'][1][:(max_len - 2)], axis=1)
                df_test['sents_ids'] = df_test.progress_apply(lambda x: x['sents_ids'][0][:(max_len - 2)], axis=1)
            save_as_pickle("df_test.pkl", df_test)
    
    logger.info("Done and saved preprocessed data!")
    return vocab, tokenizer, df_train, df_test

def preprocess_data(args):
    if not os.path.isfile("./data/df_train.pkl"):
        #df_train, df_test = get_POS_twitter_data(args, load_extracted=False)
        df_train, df_test = get_POS_conll2003_data(args, load_extracted=False)
        vocab, tokenizer, df_train, df_test = pos_preprocess(args, df_train, df_test)
    else:
        df_train = load_pickle("df_train.pkl")
        if os.path.isfile("./data/df_test.pkl"):
            df_test = load_pickle("df_test.pkl")
        else:
            df_test = None
        logger.info("Loaded preprocessed data!")
        
    return df_train, df_test
    

class Pad_Sequence():
    """
    collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
    Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0) # tokenizer.pad_token_id = 0
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) # vocab.ner2idx['<pad>'] = 0
        y_lengths = torch.LongTensor([len(x) for x in labels])
        return seqs_padded, labels_padded#, x_lengths, y_lengths

class text_dataset(Dataset):
    def __init__(self, df, args):
        
        def x_padder(x, max_len):
            if len(x) < max_len:
                x = np.array(x, dtype=int)
                x = np.append(x, np.ones((max_len-x.shape[-1]), dtype=int), axis=0)
                x = list(x)
            return x
        
        if args.model_no == 1:
            self.X = df["body"].apply(lambda x: x_padder(x, args.max_features_length))
        else:
            self.X = df["sents_ids"]
        self.y = df["poss_ids"]
        self.max_x_len = int(max(df["sents_ids"].apply(lambda x: len(x))))
        self.max_y_len = int(max(df["poss_ids"].apply(lambda x: len(x))))
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X.iloc[idx])
        y = torch.tensor(self.y.iloc[idx])
        return X, y

def load_dataloaders(args, use_other=False):
    """Load processed data if exist, else do preprocessing and loads it.  Feeds preprocessed data into dataloader, 
    returns dataloader """
    logger.info("Loading dataloaders...")
    
    if not use_other:
        df_train, df_test = preprocess_data(args)
        
        trainset = text_dataset(df_train, args)
        #max_features_length = trainset.max_x_len
        #max_seq_len = trainset.max_y_len
        train_length = len(trainset)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,\
                                  num_workers=0, collate_fn=Pad_Sequence(), pin_memory=False)
        
        if df_test is not None:
            testset = text_dataset(df_test, args)
            test_length = len(testset)
            test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=True,\
                                      num_workers=0, collate_fn=Pad_Sequence(), pin_memory=False)
        else:
            test_loader, test_length = None, None
    
    else:
        vocab = vocab_mapper()
        vocab.save()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train_loader, train_length, test_loader, test_length = load_and_cache_examples(args, tokenizer)
    
    return train_loader, train_length, test_loader, test_length
