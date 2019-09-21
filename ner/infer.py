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
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def infer(args, from_data=False):
    args.batch_size = 1
    cuda = torch.cuda.is_available()
    
    vocab = load_pickle("vocab.pkl")
        
    logger.info("NER Vocabulary size: %d" % len(vocab.ner2idx))    
    
    logger.info("Loading model and optimizers...")
    net, _, _, _, start_epoch, acc = load_model_and_optimizer(args, cuda)
    
    
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
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
        while True:
            sent = input("Type input sentence:\n")
            sent = sent.lower()
            if sent in ["quit", "exit"]:
                break
            
            if args.model_no == 0:
                sent = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)
                sent_mask = (sent != 0).long()
                if cuda:
                    sent = sent.cuda().long(); sent_mask = sent_mask.cuda()
                outputs = net(sent, attention_mask=sent_mask)
                outputs = outputs[0][:, 1:-1, :]
                o = list(torch.softmax(outputs, dim=2).max(2)[1].numpy())
                print("Sample Output: ", " ".join(vocab.idx2ner[oo] for oo in o))