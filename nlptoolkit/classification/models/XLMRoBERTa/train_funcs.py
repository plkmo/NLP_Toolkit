# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:44:38 2019

@author: WT
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .preprocessing_funcs import preprocess, load_pickle
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_dataloaders(args):
    train_path = "./data/train_processed.pkl"
    test_path = "./data/infer_processed.pkl"
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        df_train = pd.read_pickle(train_path)
        df_test = pd.read_pickle(test_path)
        logger.info("Loaded preprocessed data.")
    else:
        logger.info("Preprocessing...")
        preprocess(args)
        df_train = pd.read_pickle(train_path)
        df_test = pd.read_pickle(test_path)
        
    train_set = sentiments(df_train, tokens_length=args.tokens_length, labels=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    if args.train_test_split == 1:
        test_set = sentiments(df_test, tokens_length=args.tokens_length, labels=True)
    else:
        test_set = sentiments(df_test, tokens_length=args.tokens_length, labels=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    del df_train, df_test
    return train_loader, test_loader, len(train_set), len(test_set)

class sentiments(Dataset):
    def __init__(self, df, tokens_length=300, labels=True):
        self.X = torch.tensor(df["text"],requires_grad=False)
        self.labels = labels
        if self.labels == True:
            self.y = torch.tensor(df["label"],requires_grad=False)
        self.type = torch.zeros([len(df["text"]), tokens_length], requires_grad=False).long()
        s = torch.ones([len(df["text"]), tokens_length],requires_grad=False).long()
        for i in range(len(s)):
            if df["fills"].loc[i] != 0:
                s[i, -df["fills"].loc[i]:] = 0
        self.mask = s
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.labels == True:
            return self.X[idx], self.type[idx], self.mask[idx], self.y[idx]
        else:
            return self.X[idx], self.type[idx], self.mask[idx], 0

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
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred

def load_results(args):
    """ Loads saved results if exists """
    losses_path = "./data/test_losses_per_epoch_%d.pkl" % args.model_no
    accuracy_path = "./data/test_accuracy_per_epoch_%d.pkl" % args.model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % args.model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch

def model_eval(net, test_loader, cuda=None):
    correct = 0
    total = 0
    print("Evaluating...")
    with torch.no_grad():
        net.eval()
        for data in tqdm(test_loader):
            images, token_type, mask, labels = data
            if cuda:
                images, token_type, mask, labels = images.cuda(), token_type.cuda(), mask.cuda(), labels.cuda()
            images = images.long(); labels = labels.long()
            outputs = net(images, token_type_ids=token_type, attention_mask=mask)
            outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the %d test data points: %d %%" % (total,\
                                                                    100*correct/total))
    return 100*correct/total

def infer(infer_loader, net):
    logger.info("Evaluating on inference data...")
    cuda = next(net.parameters()).is_cuda
    net.eval()
    preds = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(infer_loader, 0), total = len(infer_loader)):
            inputs, token_type, mask, _ = data
            if cuda:
                inputs, token_type, mask = inputs.cuda(), token_type.cuda(), mask.cuda()
            inputs = inputs.long()
            outputs = net(inputs, token_type_ids=token_type, attention_mask=mask)
            outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            predicted = list(predicted.cpu().numpy()) if cuda else list(predicted.numpy())
            preds.extend(predicted)
            
    df_results = pd.DataFrame(columns=["index", "predicted_label"])
    df_results.loc[:, "index"] = [i for i in range(len(preds))]
    df_results.loc[:, "predicted_label"] = preds
    df_results.to_csv("./data/results.csv", columns=df_results.columns, index=False)
    return df_results