# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 09:17:59 2019

@author: WT
"""
import torch
from torch.autograd import Variable
from .models.BERT.tokenization_bert import BertTokenizer
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_model_and_optimizer, decode_outputs
from .utils.misc_utils import load_pickle
from tqdm import tqdm
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class infer_from_trained(object):
    def __init__(self, args=None):
        
        logger.info("Loading model and vocab...")
        if args is not None:
            self.args = args
        else:
            self.args = load_pickle("./data/args.pkl")
        
        self.args.batch_size = 1
        self.cuda = torch.cuda.is_available()
        self.vocab = load_pickle("vocab.pkl")
        
        logger.info("NER Vocabulary size: %d" % (len(self.vocab.ner2idx) - 1))
        net, _, _, _, start_epoch, acc = load_model_and_optimizer(self.args, 10, self.cuda)
        self.net = net
        self.max_len = self.args.tokens_length - 2
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.cls_id = self.tokenizer.convert_tokens_to_ids(["[CLS]"])
        self.sep_id = self.tokenizer.convert_tokens_to_ids(["[SEP]"])
    
    def infer_from_data(self,):
        self.net.eval()
        train_loader, train_length, test_loader, test_length = load_dataloaders(self.args)
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                
                if self.args.model_no == 0:
                    if len(data) == 4:
                        src_input = data[0]
                        src_mask = data[1]
                        token_type = data[2]
                        labels = data[3]
                    else:
                        src_input = data[0]
                        labels = data[1]
                        src_mask = (src_input != 0).long()
                        token_type = torch.zeros((src_input.shape[0], src_input.shape[1]), dtype=torch.long)
                    if self.cuda:
                        src_input = src_input.cuda().long(); labels = labels.cuda().long()
                        src_mask = src_mask.cuda(); token_type=token_type.cuda()
                    outputs = self.net(src_input, attention_mask=src_mask, token_type_ids=token_type)
                    outputs = outputs[0]
                    
                elif self.args.model_no == 1:
                    src_input, trg_input = data[0], data[1][:, :-1]
                    labels = data[1][:,1:].contiguous().view(-1)
                    if self.cuda:
                        src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    outputs = self.net(src_input, trg_input)
                
                decode_outputs(outputs, labels, self.vocab.idx2ner, self.args, reshaped=False)
                print("")
                time.sleep(7)
    
    def infer_from_input(self,):
        self.net.eval()
        while True:
            sent = input("Type input sentence: (\'quit\' or \'exit\' to terminate)\n")
            sent = sent.lower()
            if sent in ["quit", "exit"]:
                break
            words = sent.split()
            if self.args.model_no == 0:
                sent = torch.tensor(self.cls_id + self.tokenizer.encode(sent) + self.sep_id).unsqueeze(0)
                sent_mask = (sent != 0).long()
                token_type = torch.zeros((sent.shape[0], sent.shape[1]), dtype=torch.long)
                if self.cuda:
                    sent = sent.cuda().long(); sent_mask = sent_mask.cuda()
                    token_type=token_type.cuda()
                outputs = self.net(sent, attention_mask=sent_mask, token_type_ids=token_type)
                outputs = outputs[0]
                o = torch.softmax(outputs, dim=2).max(2)[1].cpu().numpy().tolist()[0] if outputs.is_cuda else\
                         torch.softmax(outputs, dim=2).max(2)[1].numpy().tolist()[0]
                decoded = [self.vocab.idx2ner[oo] for oo in o[1:-1]]
                
                pointer = 0; ner_tags = []
                for word in words:
                    ner_tags.append(decoded[pointer])
                    pointer += len(self.tokenizer.tokenize(word))
                
                assert len(ner_tags) == len(words)
                
                print("Words --- Tags:")
                for word, tag in zip(words, ner_tags):
                    print("%s (%s) " % (word, tag))
    
    def infer_from_file(self, in_file, out_file):
        
        logger.info("Reading file %s..." % in_file)
        with open(in_file, "r", encoding="utf8") as f:
            sents = f.readlines()
        
        logger.info("Tagging...")
        self.net.eval()
        sents_tags = []
        for sent in tqdm(sents):
            if self.args.model_no == 0:
                sent = sent.lower()
                words = self.tokenizer.tokenize(sent)
                sent = torch.tensor(self.cls_id + self.tokenizer.encode(sent) + self.sep_id).unsqueeze(0)
                sent_mask = (sent != 0).long()
                token_type = torch.zeros((sent.shape[0], sent.shape[1]), dtype=torch.long)
                if self.cuda:
                    sent = sent.cuda().long(); sent_mask = sent_mask.cuda()
                    token_type=token_type.cuda()
                outputs = self.net(sent, attention_mask=sent_mask, token_type_ids=token_type)
                outputs = outputs[0]
                o = torch.softmax(outputs, dim=2).max(2)[1].cpu().numpy().tolist()[0] if outputs.is_cuda else\
                         torch.softmax(outputs, dim=2).max(2)[1].numpy().tolist()[0]
                decoded = [self.vocab.idx2ner[oo] for oo in o[1:-1]]
                
                pointer = 0; ner_tags = []
                for word in words:
                    ner_tags.append(decoded[pointer])
                    pointer += len(self.tokenizer.encode(word)) 
                
                assert len(ner_tags) == len(words)
                
                sents_tags.append(" ".join(ner_tags))
                
        logger.info("Saving to %s..." % out_file)
        with open(out_file, "w", encoding="utf8") as f:
            for sent_tags in tqdm(sents_tags):
                f.write(sent_tags + "\n")
        logger.info("Done!")
        

def infer(args, from_data=False):
    args.batch_size = 1
    cuda = torch.cuda.is_available()
    
    vocab = load_pickle("vocab.pkl")
        
    logger.info("NER Vocabulary size: %d" % (len(vocab.ner2idx) - 1))    
    
    logger.info("Loading model and optimizers...")
    net, _, _, _, start_epoch, acc = load_model_and_optimizer(args, 10, cuda)

    net.eval()
    
    if from_data:
        train_loader, train_length, test_loader, test_length = load_dataloaders(args)
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                
                if args.model_no == 0:
                    if len(data) == 4:
                        src_input = data[0]
                        src_mask = data[1]
                        token_type = data[2]
                        labels = data[3]
                    else:
                        src_input = data[0]
                        labels = data[1]
                        src_mask = (src_input != 0).long()
                        token_type = torch.zeros((src_input.shape[0], src_input.shape[1]), dtype=torch.long)
                    if cuda:
                        src_input = src_input.cuda().long(); labels = labels.cuda().long()
                        src_mask = src_mask.cuda(); token_type=token_type.cuda()
                    outputs = net(src_input, attention_mask=src_mask, token_type_ids=token_type)
                    outputs = outputs[0]
                    
                elif args.model_no == 1:
                    src_input, trg_input = data[0], data[1][:, :-1]
                    labels = data[1][:,1:].contiguous().view(-1)
                    if cuda:
                        src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    outputs = net(src_input, trg_input)
                
                #print(outputs.shape); print(labels.shape)
                #outputs = outputs.reshape(-1, outputs.size(-1))
                #outputs = outputs.view(-1, outputs.size(-1))
                
                decode_outputs(outputs, labels, vocab.idx2ner, args, reshaped=False)
                print("")
                time.sleep(7)
    else:
        max_len = args.tokens_length - 2
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        cls_id = tokenizer.convert_tokens_to_ids(["[CLS]"])
        sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])
        while True:
            sent = input("Type input sentence: (\'quit\' or \'exit\' to terminate)\n")
            sent = sent.lower()
            if sent in ["quit", "exit"]:
                break
            
            if args.model_no == 0:
                words = tokenizer.tokenize(sent)
                sent = torch.tensor(cls_id + tokenizer.encode(sent) + sep_id).unsqueeze(0)
                sent_mask = (sent != 0).long()
                token_type = torch.zeros((sent.shape[0], sent.shape[1]), dtype=torch.long)
                if cuda:
                    sent = sent.cuda().long(); sent_mask = sent_mask.cuda()
                    token_type=token_type.cuda()
                outputs = net(sent, attention_mask=sent_mask, token_type_ids=token_type)
                outputs = outputs[0]
                o = torch.softmax(outputs, dim=2).max(2)[1].cpu().numpy().tolist()[0] if outputs.is_cuda else\
                         torch.softmax(outputs, dim=2).max(2)[1].numpy().tolist()[0]
                decoded = [vocab.idx2ner[oo] for oo in o[1:-1]]
                
                pointer = 0; ner_tags = []
                for word in words:
                    ner_tags.append(decoded[pointer])
                    pointer += len(tokenizer.encode(word)) 
                
                assert len(ner_tags) == len(words)
                
                print("Words --- Tags:")
                for word, tag in zip(words, ner_tags):
                    print("%s (%s) " % (word, tag))