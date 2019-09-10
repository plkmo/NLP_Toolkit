# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:26:35 2019

@author: tsd
"""
import torch
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_model_and_optimizer
from .utils.bpe_vocab import Encoder
from .utils.misc import save_as_pickle, load_pickle
from .utils.word_char_level_vocab import tokener
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class trg2_vocab_obj(object):
    def __init__(self, idx_mappings, mappings):
        map2 = {}
        for punc in mappings.keys():
            map2[punc] = idx_mappings[mappings[punc]]
        map2['word'] = len(map2)
        map2['sos'] = len(map2)
        map2['eos'] = len(map2)
        map2['pad'] = len(map2)
        self.punc2idx = map2
        self.idx2punc = {v:k for k,v in map2.items()}

def infer(args, from_data=False):
    
    args.batch_size = 1
    cuda = torch.cuda.is_available()
    
    if args.model_no == 0:
        from .models.Transformer import create_masks, create_trg_mask
    elif args.model_no == 2:
        from .models.py_Transformer import create_masks, create_trg_mask
    
    if args.level == "bpe":
        logger.info("Loading BPE info...")
        vocab = Encoder.load("./data/vocab.pkl")
        vocab_size = len(vocab.bpe_vocab) + len(vocab.word_vocab)
        tokenizer_en = tokener("en")
        vocab.word_tokenizer = tokenizer_en.tokenize
        vocab.custom_tokenizer = True
        mappings = load_pickle("mappings.pkl") # {'!': 250, '?': 34, '.': 5, ',': 4}
        idx_mappings = load_pickle("idx_mappings.pkl") # {250: 0, 34: 1, 5: 2, 4: 3, 'word': 4, 'sos': 5, 'eos': 6, 'pad': 7}
    
    trg2_vocab = trg2_vocab_obj(idx_mappings, mappings)
    
    logger.info("Loading model and optimizers...")
    net, _, _, _, _, _ = load_model_and_optimizer(args=args, src_vocab_size=vocab_size, \
                                                                                      trg_vocab_size=vocab_size,\
                                                                                      trg2_vocab_size=len(idx_mappings),\
                                                                                      max_features_length=args.max_encoder_len,\
                                                                                      max_seq_length=args.max_decoder_len, \
                                                                                      mappings=mappings,\
                                                                                      idx_mappings=idx_mappings,\
                                                                                      cuda=cuda)
    if from_data:
        _, train_loader, train_length, max_features_length, max_output_len = load_dataloaders(args)
        with torch.no_grad():
            for i, data in enumerate(train_loader):
            
                if args.model_no == 0:
                    src_input, trg_input, trg2_input = data[0], data[1][:, :-1], data[2][:, :-1]
                    #labels = data[1][:,1:].contiguous().view(-1)
                    #labels2 = data[2][:,1:].contiguous().view(-1)
                    src_mask, trg_mask = create_masks(src_input, trg_input)
                    trg2_mask = create_trg_mask(trg2_input, ignore_idx=idx_mappings['pad'])
                    if cuda:
                        src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); #labels = labels.cuda().long()
                        src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda(); trg2_mask = trg2_mask.cuda()
                        trg2_input = trg2_input.cuda().long(); #labels2 = labels2.cuda().long()
                    # self, src, trg, trg2, src_mask, trg_mask=None, trg_mask2=None, infer=False, trg_vocab_obj=None, \
                    #trg2_vocab_obj=None
                    stepwise_translated_words, final_step_words, stepwise_translated_words2, final_step_words2 = net(src_input, \
                                                                                                                     trg_input[:,0].unsqueeze(0), \
                                                                                                                     trg2_input[:,0].unsqueeze(0),\
                                                                                                                     src_mask, \
                                                                                                                     trg_mask, \
                                                                                                                     trg2_mask, \
                                                                                                                     True, \
                                                                                                                     vocab, \
                                                                                                                     trg2_vocab)
                    print("\nStepwise-translated:")
                    print(" ".join(stepwise_translated_words))
                    print()
                    print("\nFinal step translated words: ")
                    print(" ".join(final_step_words))
                    print()
                    print("\nStepwise-translated2:")
                    print(" ".join(stepwise_translated_words2))
                    print()
                    print("\nFinal step translated words2: ")
                    print(" ".join(final_step_words2))
                    print()
                    time.sleep(10)
                
    else:
        while True:
            with torch.no_grad():
                sent = input("Input sentence to punctuate:\n")
                if sent in ["quit", "exit"]:
                    break
                sent = torch.tensor(next(vocab.transform([sent]))).unsqueeze(0)
                trg_input = torch.tensor([vocab.word_vocab['__sos']]).unsqueeze(0)
                trg2_input = torch.tensor([idx_mappings['sos']]).unsqueeze(0)
                src_mask, trg_mask = create_masks(sent, trg_input)
                trg2_mask = create_trg_mask(trg2_input, ignore_idx=idx_mappings['pad'])
                if cuda:
                    sent = sent.cuda().long(); trg_input = trg_input.cuda().long(); trg2_input = trg2_input.cuda().long()
                    src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda(); trg2_mask = trg2_mask.cuda()
                stepwise_translated_words, final_step_words, stepwise_translated_words2, final_step_words2 = net(sent, \
                                                                                                                             trg_input, \
                                                                                                                             trg2_input,\
                                                                                                                             src_mask, \
                                                                                                                             trg_mask, \
                                                                                                                             trg2_mask, \
                                                                                                                             True, \
                                                                                                                             vocab, \
                                                                                                                             trg2_vocab)
                
                print("\nStepwise-translated:")
                print(" ".join(stepwise_translated_words))
                print()
                print("\nFinal step translated words: ")
                print(" ".join(final_step_words))
                print()
                print("\nStepwise-translated2:")
                print(" ".join(stepwise_translated_words2))
                print()
                print("\nFinal step translated words2: ")
                print(" ".join(final_step_words2))
                print()

    return

class punctuator(object):
    def __init__(self,):
        super(punctuator, self).__init__()
        logger.info("Loading data...")
        self.args = load_pickle("args.pkl")
        
        if self.args.model_no == 0:
            from .models.Transformer import create_masks, create_trg_mask
        elif self.args.model_no == 2:
            from .models.py_Transformer import create_masks, create_trg_mask
        else:
            create_masks, create_trg_mask = None, None
            
        self.create_masks = create_masks
        self.create_trg_mask = create_trg_mask
        self.args.batch_size = 1
        self.cuda = torch.cuda.is_available()
        if self.args.level == "bpe":
            self.vocab = Encoder.load("./data/vocab.pkl")
            self.vocab_size = len(self.vocab.bpe_vocab) + len(self.vocab.word_vocab)
            self.tokenizer_en = tokener("en")
            self.vocab.word_tokenizer = self.tokenizer_en.tokenize
            self.vocab.custom_tokenizer = True
            self.mappings = load_pickle("mappings.pkl") # {'!': 250, '?': 34, '.': 5, ',': 4}
            self.idx_mappings = load_pickle("idx_mappings.pkl") # {250: 0, 34: 1, 5: 2, 4: 3, 'word': 4, 'sos': 5, 'eos': 6, 'pad': 7}
    
        self.trg2_vocab = trg2_vocab_obj(self.idx_mappings, self.mappings)
        
        logger.info("Loading model and optimizers...")
        self.net, _, _, _, _, _ = load_model_and_optimizer(args=self.args, src_vocab_size=self.vocab_size, \
                                                                                          trg_vocab_size=self.vocab_size,\
                                                                                          trg2_vocab_size=len(self.idx_mappings),\
                                                                                          max_features_length=self.args.max_encoder_len,\
                                                                                          max_seq_length=self.args.max_decoder_len, \
                                                                                          mappings=self.mappings,\
                                                                                          idx_mappings=self.idx_mappings,\
                                                                                          cuda=self.cuda)
    
    def punctuate(self, sent):
        sent = torch.tensor(next(self.vocab.transform([sent]))).unsqueeze(0)
        if self.args.level == "bpe":
            trg_input = torch.tensor([self.vocab.word_vocab['__sos']]).unsqueeze(0)
            trg2_input = torch.tensor([self.idx_mappings['sos']]).unsqueeze(0)
        
        if self.args.model_no == 0:
            src_mask, trg_mask = self.create_masks(sent, trg_input)
            trg2_mask = self.create_trg_mask(trg2_input, ignore_idx=self.idx_mappings['pad'])
            if self.cuda:
                sent = sent.cuda().long(); trg_input = trg_input.cuda().long(); trg2_input = trg2_input.cuda().long()
                src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda(); trg2_mask = trg2_mask.cuda()
            with torch.no_grad():
                stepwise_translated_words, final_step_words, stepwise_translated_words2, final_step_words2 = self.net(sent, \
                                                                                                                 trg_input, \
                                                                                                                 trg2_input,\
                                                                                                                 src_mask, \
                                                                                                                 trg_mask, \
                                                                                                                 trg2_mask, \
                                                                                                                 True, \
                                                                                                                 self.vocab, \
                                                                                                                 self.trg2_vocab)
        return sent, stepwise_translated_words2