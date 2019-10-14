# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 12:26:35 2019

@author: tsd
"""
import re
import pandas as pd
import torch
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_model_and_optimizer
from .utils.bpe_vocab import Encoder
from .utils.misc import save_as_pickle, load_pickle
from .utils.word_char_level_vocab import tokener
from tqdm import tqdm
import time
import logging

tqdm.pandas(desc="prog-bar")
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

def find(s, ch=[".", "!", "?"]):
    return [i for i, ltr in enumerate(s) if ltr in ch]

def corrector_module(corrected, sent=None, cap_abbrev=True):
    corrected = corrected[0].upper() + corrected[1:]
    corrected = re.sub(r" +([\.\?!,])", r"\1", corrected) # corrected = re.sub(" +[,]", ",", corrected)
    corrected = re.sub("' +", "'", corrected) # remove extra spaces from ' s
    idxs = find(corrected)
    for idx in idxs:
        if (idx + 3) < len(corrected):
            corrected = corrected[:idx + 2] + corrected[(idx + 2)].upper() + corrected[(idx + 3):]
            
    if cap_abbrev == True:
        abbrevs = ["ntu", "nus", "smrt", "sutd", "sim", "smu", "i2r", "astar", "imda", "hdb", "edb", "lta", "cna",\
                   "suss"]
        corrected = corrected.split()
        corrected1 = []
        for word in corrected:
            if word.lower().strip("!?.,") in abbrevs:
                corrected1.append(word.upper())
            else:
                corrected1.append(word)
        assert len(corrected) == len(corrected1)
        corrected = " ".join(corrected1)
    return corrected

class infer_from_trained(object):
    def __init__(self, args=None):
        
        logger.info("Loading model and vocab...")
        if args is not None:
            self.args = args
        else:
            self.args = load_pickle("args.pkl")
        
        self.args.batch_size = 1
        self.cuda = torch.cuda.is_available()
        
        if self.args.model_no == 0:
            from .models.Transformer import create_masks, create_trg_mask
        elif self.args.model_no == 2:
            from .models.py_Transformer import create_masks, create_trg_mask
        
        if self.args.model_no != 1:
            self.create_masks = create_masks
            self.create_trg_mask = create_trg_mask
        
        if self.args.level == "bpe":
            logger.info("Loading BPE info...")
            self.vocab = Encoder.load("./data/vocab.pkl")
            self.vocab_size = len(self.vocab.bpe_vocab) + len(self.vocab.word_vocab)
            self.tokenizer_en = tokener("en")
            self.vocab.word_tokenizer = self.tokenizer_en.tokenize
            self.vocab.custom_tokenizer = True
            self.mappings = load_pickle("mappings.pkl") # {'!': 250, '?': 34, '.': 5, ',': 4}
            self.idx_mappings = load_pickle("idx_mappings.pkl") # {250: 0, 34: 1, 5: 2, 4: 3, 'word': 4, 'sos': 5, 'eos': 6, 'pad': 7}
        
        self.trg2_vocab = trg2_vocab_obj(self.idx_mappings, self.mappings)
        
        logger.info("Loading model and optimizers...")
        net, _, _, _, _, _ = load_model_and_optimizer(args=self.args, src_vocab_size=self.vocab_size, \
                                                                                          trg_vocab_size=self.vocab_size,\
                                                                                          trg2_vocab_size=len(self.idx_mappings),\
                                                                                          max_features_length=self.args.max_encoder_len,\
                                                                                          max_seq_length=self.args.max_decoder_len, \
                                                                                          mappings=self.mappings,\
                                                                                          idx_mappings=self.idx_mappings,\
                                                                                          cuda=self.cuda)
        self.net = net
        self.net.eval()
        
    def infer_from_data(self,):
        _, train_loader, train_length, max_features_length, max_output_len = load_dataloaders(self.args)
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(train_loader):
            
                if self.args.model_no == 0:
                    src_input, trg_input, trg2_input = data[0], data[1][:, :-1], data[2][:, :-1]
                    src_mask, trg_mask = self.create_masks(src_input, trg_input)
                    trg2_mask = self.create_trg_mask(trg2_input, ignore_idx=self.idx_mappings['pad'])
                    if self.cuda:
                        src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long();
                        src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda(); trg2_mask = trg2_mask.cuda()
                        trg2_input = trg2_input.cuda().long(); 
                    stepwise_translated_words, final_step_words, stepwise_translated_words2, final_step_words2 = self.net(src_input, \
                                                                                                                     trg_input[:,0].unsqueeze(0), \
                                                                                                                     trg2_input[:,0].unsqueeze(0),\
                                                                                                                     src_mask, \
                                                                                                                     trg_mask, \
                                                                                                                     trg2_mask, \
                                                                                                                     True, \
                                                                                                                     self.vocab, \
                                                                                                                     self.trg2_vocab)
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
                
                elif self.args.model_no == 1:
                    src_input, trg_input, trg2_input = data[0], data[1][:, :-1], data[2][:, :-1]
                    labels = data[1][:,1:].contiguous().view(-1)
                    labels2 = data[2][:,1:].contiguous().view(-1)
                    if self.cuda:
                        src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                        trg2_input = trg2_input.cuda().long(); labels2 = labels2.cuda().long()
                    outputs, outputs2 = self.net(src_input, trg_input, trg2_input, infer=True)
                    outputs2 = outputs2.cpu().numpy().tolist() if outputs2.is_cuda else outputs2.cpu().numpy().tolist()
                    punc = [self.trg2_vocab.idx2punc[i] for i in outputs2[0]]
                    print(punc)
                    punc = [self.mappings[p] if p in ['!', '?', '.', ','] else p for p in punc]
                    l = list(labels[:70].cpu().numpy()) if labels.is_cuda else list(labels[:70].numpy())
                    l = [l] if self.args.level == "bpe" else l
                    print("Sample Label: ", " ".join(self.vocab.inverse_transform(l)))
                    
                    src_input = src_input[src_input != 1]
                    src_input = src_input.cpu().numpy().tolist() if self.cuda else src_input.numpy().tolist()
                    counter = 0
                    for idx, p in enumerate(punc):
                        if (p == 'word') and (counter < len(src_input)):
                            punc[idx] = src_input[counter]
                            counter += 1
                        elif (p == "eos"):
                            break
                        elif (counter >= len(src_input)) and (p in ['word', 'sos']):
                            punc[idx] = 5
                            break

                    print("Predicted Label: ", " ".join(self.vocab.inverse_transform([punc[:idx]])))

                    time.sleep(10)
                    
    def infer_sentence(self, sentence):
        sent = torch.tensor(next(self.vocab.transform([sentence]))).unsqueeze(0)
        if self.args.model_no == 0:
            trg_input = torch.tensor([self.vocab.word_vocab['__sos']]).unsqueeze(0)
            trg2_input = torch.tensor([self.idx_mappings['sos']]).unsqueeze(0)
            src_mask, trg_mask = self.create_masks(sent, trg_input)
            trg2_mask = self.create_trg_mask(trg2_input, ignore_idx=self.idx_mappings['pad'])
            if self.cuda:
                sent = sent.cuda().long(); trg_input = trg_input.cuda().long(); trg2_input = trg2_input.cuda().long()
                src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda(); trg2_mask = trg2_mask.cuda()
            stepwise_translated_words, final_step_words, stepwise_translated_words2, final_step_words2 = self.net(sent, \
                                                                                                                         trg_input, \
                                                                                                                         trg2_input,\
                                                                                                                         src_mask, \
                                                                                                                         trg_mask, \
                                                                                                                         trg2_mask, \
                                                                                                                         True, \
                                                                                                                         self.vocab, \
                                                                                                                         self.trg2_vocab)
            
            step_ = " ".join(stepwise_translated_words)
            final_ = " ".join(final_step_words)
            if not (step_ == '') or not (final_ == ''):
                step_ = corrector_module(step_)
                final_ = corrector_module(final_)
                final_2 = " ".join(final_step_words2)
                print("\nStepwise-translated:")
                print(step_)
                print()
                print("\nFinal step translated words: ")
                print(final_)
                print()
                print("\nStepwise-translated2:")
                print(" ".join(stepwise_translated_words2))
                print()
                print("\nFinal step translated words2: ")
                print(final_2)
                print()
                predicted = step_
            else:
                print("None, please try another sentence.")
                predicted = "None"
            
        elif self.args.model_no == 1:
            sent = torch.nn.functional.pad(sent,[0, (self.args.max_encoder_len - sent.shape[1])], value=1)
            trg_input = torch.tensor([self.vocab.word_vocab['__sos']]).unsqueeze(0)
            trg2_input = torch.tensor([self.idx_mappings['sos']]).unsqueeze(0)
            if self.cuda:
                sent = sent.cuda().long(); trg_input = trg_input.cuda().long()
                trg2_input = trg2_input.cuda().long()
            outputs, outputs2 = self.net(sent, trg_input, trg2_input, infer=True)
            outputs2 = outputs2.cpu().numpy().tolist() if outputs2.is_cuda else outputs2.cpu().numpy().tolist()
            punc = [self.trg2_vocab.idx2punc[i] for i in outputs2[0]]
            #print(punc)
            punc = [self.mappings[p] if p in ['!', '?', '.', ','] else p for p in punc]
            
            sent = sent[sent != 1]
            sent = sent.cpu().numpy().tolist() if self.cuda else sent.numpy().tolist()
            counter = 0
            for idx, p in enumerate(punc):
                if (p == 'word') and (counter < len(sent)):
                    punc[idx] = sent[counter]
                    counter += 1
                elif (p == "eos"):
                    break
                elif (counter >= len(sent)) and (p in ['word', 'sos']):
                    punc[idx] = 5
                    break
            
            predicted = " ".join(self.vocab.inverse_transform([punc[:idx]]))
            predicted = corrector_module(predicted)
            print("Predicted Label: ", predicted)
        return predicted
    
    def infer_from_input(self,):
        self.net.eval()
        while True:
            with torch.no_grad():
                sent = input("Input sentence to punctuate:\n")
                if sent in ["quit", "exit"]:
                    break
                predicted = self.infer_sentence(sent)
        return predicted
    
    def infer_from_file(self, in_file="./data/input.txt", out_file="./data/output.txt"):
        df = pd.read_csv(in_file, header=None, names=["sents"])
        df['labels'] = df.progress_apply(lambda x: self.infer_sentence(x['sents']), axis=1)
        df.to_csv(out_file, index=False)
        logger.info("Done and saved as %s!" % out_file)
        return

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
                
                elif args.model_no == 1:
                    src_input, trg_input, trg2_input = data[0], data[1][:, :-1], data[2][:, :-1]
                    labels = data[1][:,1:].contiguous().view(-1)
                    labels2 = data[2][:,1:].contiguous().view(-1)
                    if cuda:
                        src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                        trg2_input = trg2_input.cuda().long(); labels2 = labels2.cuda().long()
                    outputs, outputs2 = net(src_input, trg_input, trg2_input, infer=True)
                    print(outputs, outputs2)
                
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