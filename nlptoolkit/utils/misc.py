# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:46:13 2019

@author: WT
"""
import os
import pickle

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
def split_dataset(file, ratio=0.9, train='./data/train.txt', test='./data/test.txt'):
    if os.path.isfile(train) and os.path.isfile(test):
        print("Split dataset already exists!")
        return
    
    print("Reading file...")
    with open(file, 'r', encoding='utf8') as f:
        text = f.readlines()
    length = len(text)
    split_idx = int(ratio*length)
    train_text = text[:split_idx]
    test_text = text[split_idx:]
    
    with open(train, 'w', encoding='utf8') as f:
        f.writelines(train_text)
        
    with open(test, 'w', encoding='utf8') as f:
        f.writelines(test_text)
    
    print("Train sample: ", train_text[0])
    print("\nTest sample: ", test_text[0])
    print('Train length: ', len(train_text))
    print("Test length: ", len(test_text))
    print("Done and saved to %s, %s" % (train, test))
    return train_text, test_text