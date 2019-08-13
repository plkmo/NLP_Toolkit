# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 09:17:59 2019

@author: WT
"""
import torch
from torch.autograd import Variable
from .preprocessing_funcs import load_dataloaders
from .models.Transformer.transformer_model import create_masks
from .train_funcs import load_model_and_optimizer
from .utils.bpe_vocab import Encoder
from .utils import load_pickle
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def infer(args, from_data=False):
    args.batch_size = 1
    cuda = torch.cuda.is_available()
    train_loader, train_length, max_features_length, max_seq_len = load_dataloaders(args)
    
    if (args.level == "word") or (args.level == "char"):
        vocab = load_pickle("vocab.pkl")
        vocab_size = len(vocab.w2idx)
        trg_init = vocab.w2idx["<sos>"]
    elif args.level == "bpe":
        vocab = Encoder.load("./data/vocab.pkl")
        vocab_size = vocab.vocab_size
        trg_init = vocab.word_vocab["__sos"]
    trg_init = Variable(torch.LongTensor([trg_init])).unsqueeze(0)
        
    logger.info("Max features length = %d %ss" % (max_features_length, args.level))
    logger.info("Vocabulary size: %d" % vocab_size)
    
    logger.info("Loading model and optimizers...")
    net, criterion, optimizer, scheduler, start_epoch, acc = load_model_and_optimizer(args, vocab_size, max_features_length,\
                                                                                      max_seq_len, cuda)
    
    
    if from_data:
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                
                if args.model_no == 0:
                    src_input, trg_input = data[0], data[1][:, :-1]
                    labels = data[1][:,1:].contiguous().view(-1)
                    src_mask, trg_mask = create_masks(src_input, trg_input)
                    if cuda:
                        src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                        src_mask = src_mask.cuda(); trg_mask = trg_mask.cuda()
                    outputs = net(src_input, trg_input, src_mask, trg_mask)
                    
                elif args.model_no == 1:
                    src_input, trg_input = data[0], data[1][:, :-1]
                    labels = data[1][:,1:].contiguous().view(-1)
                    if cuda:
                        src_input = src_input.cuda().long(); trg_input = trg_input.cuda().long(); labels = labels.cuda().long()
                    outputs = net(src_input, trg_input, infer=True)
                outputs = outputs.view(-1, outputs.size(-1))
                
                if (args.level == "word") or (args.level == "char"):
                    vocab_decoder = vocab.convert_idx2w
                elif args.level == "bpe":
                    vocab_decoder = vocab.inverse_transform
                
                if cuda:
                    l = list(labels.cpu().numpy())
                    o = list(torch.softmax(outputs, dim=1).max(1)[1].cpu().numpy())
                else:
                    l = list(labels.numpy())
                    o = list(torch.softmax(outputs, dim=1).max(1)[1].numpy())
                if args.level == "bpe":
                    l = [l]
                    o = [o]
                print("Sample Output: ", " ".join(vocab_decoder(o)))
                print("Sample Label: ", " ".join(vocab_decoder(l)))
                time.sleep(7)
    else:
        pass