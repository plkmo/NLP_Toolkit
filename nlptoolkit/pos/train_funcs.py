# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:23:00 2019

@author: WT
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from .models.BERT.modeling_bert import BertForTokenClassification
from .models.BERT.configuration_bert import BertConfig
from .utils.misc_utils import load_pickle, save_as_pickle, CosineWithRestarts
from .models.optimization import AdamW, WarmupLinearSchedule
from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_model_and_optimizer(args, len_train_loader, cuda=False):
    '''Loads the model (Transformer or encoder-decoder) based on provided arguments and parameters'''
    
    if args.model_no == 0:
        logger.info("Loading pre-trained BERT for token classification...")
        config = BertConfig.from_pretrained('bert-base-uncased',
                                          num_labels=args.num_classes)
        net = BertForTokenClassification.from_pretrained('bert-base-uncased', config=config)

    if cuda:
        net.cuda()
        
    criterion = nn.CrossEntropyLoss() # ignore padding tokens
    
    # Prepare optimizer and schedule (linear warmup and decay)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_epochs = args.max_steps // (len_train_loader // args.gradient_acc_steps) + 1
    else:
        t_total = len_train_loader // args.gradient_acc_steps*args.num_epochs
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    
    start_epoch, acc = load_state(net, optimizer, scheduler, args, load_best=False)

    return net, criterion, optimizer, scheduler, start_epoch, acc

def load_state(net, optimizer, scheduler, args, load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred

def load_state1(net, args, load_best=False, load_scheduler=False):
    """ Loads saved model and optimizer states if exists """
    base_path = "./data/"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % args.model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % args.model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        if load_best:
            net = net.load_model(best_path)
        else:
            net = net.load_model(checkpoint_path)
        optimizer = optim.Adam([{"params":net.bert.parameters(),"lr": args.lr/5},\
                             {"params":net.classifier.parameters(), "lr": args.lr}])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,10,13,15,17,20,23,25], gamma=0.8)
        if load_scheduler:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    else:
        optimizer = optim.Adam([{"params":net.bert.parameters(),"lr": args.lr/5},\
                             {"params":net.classifier.parameters(), "lr": args.lr}])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,10,13,15,17,20,23,25], gamma=0.8)
    return net, optimizer, scheduler, start_epoch, best_pred

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./data/test_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch

def evaluate(output, labels, ignore_idx):
    ### ignore index 0 (padding) when calculating accuracy
    idxs = (labels != ignore_idx).nonzero().squeeze()
    o_labels = torch.softmax(output, dim=1).max(1)[1]; #print(output.shape, o_labels.shape)
    l = labels[idxs]; o = o_labels[idxs]
    
    if len(idxs) > 1:
        acc = (l == o).sum().item()/len(idxs)
    else:
        acc = (l == o).sum().item()
    l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
    o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()
    return acc, (o, l)

def evaluate_results(net, data_loader, cuda, g_mask1, g_mask2, args, ignore_idx, idx2pos):
    acc = 0
    print("Evaluating...")
    out_labels = []; true_labels = []
    with torch.no_grad():
        net.eval()
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            if args.model_no == 0:
                if len(data) == 4:
                    src_input = data[0]
                    src_mask = data[1]
                    token_type = data[2]
                    labels = data[3].contiguous().view(-1)
                else:
                    src_input = data[0]
                    labels = data[1].contiguous().view(-1)
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
            outputs = outputs.reshape(-1, outputs.size(-1))
            cal_acc, (o, l) = evaluate(outputs, labels, ignore_idx)
            out_labels.append([idx2pos[i] for i in o]); true_labels.append([idx2pos[i] for i in l])
            acc += cal_acc
            
    eval_acc = acc/(i + 1)
    results = {
        "accuracy": eval_acc,
        "precision": precision_score(true_labels, out_labels),
        "recall": recall_score(true_labels, out_labels),
        "f1": f1_score(true_labels, out_labels)
    }

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
        
    return results

def decode_outputs(outputs, labels, vocab_decoder, args, reshaped=True):
    if reshaped:
        if labels.is_cuda:
            l = list(labels[:50].cpu().numpy())
            o = list(torch.softmax(outputs, dim=1).max(1)[1][:50].cpu().numpy())
        else:
            l = list(labels[:50].numpy())
            o = list(torch.softmax(outputs, dim=1).max(1)[1][:50].numpy())
        
        print("Sample Output: ", " ".join(vocab_decoder[oo] for oo in o))
        print("Sample Label: ", " ".join(vocab_decoder[ll] for ll in l))
    
    else:
        if labels.is_cuda:
            l = labels[0,:].cpu().numpy().tolist()
            o = torch.softmax(outputs, dim=2).max(2)[1].cpu().numpy().tolist()[0]
        else:
            l = labels[0,:].numpy().tolist()
            o = torch.softmax(outputs, dim=2).max(2)[1].numpy().tolist()[0]
        print("Sample Output: ", " ".join(vocab_decoder[oo] for oo in o))
        print("Sample Label: ", " ".join(vocab_decoder[ll] for ll in l))
