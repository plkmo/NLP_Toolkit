# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:53:42 2019

@author: WT
"""

import os
import librosa
import soundfile as sf
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from .utils import save_as_pickle, load_pickle, tokener, vocab
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

class args():
    def __init__(self):
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        self.batch_size = 5
        self.use_lg_mels = 1
        self.n_mels = 80
        self.max_frame_len = 3171
        
def separate_info(x):
    id_ = re.search("[\d+-]+", x)[0]
    speaker = re.search("[\d+]+-", x)[0][:-1]
    text = re.search("([A-Z]+.+)+", x)[0]
    return id_, speaker, text

def pad_sos_eos(x):
    return [0] + x + [2]

def normal_scaler(x, mu, std):
    return (x-mu)/std

def min_max_scaler(x, min_, max_, feature_range=[0.3,2.3]):
    x_std = (x - min_)/(max_- min_)
    x_scaled = x_std*(feature_range[1] - feature_range[0]) + feature_range[0]
    return x_scaled

def stack_downsampler(feature, args, stack=3):
    '''stack: number of frames to stack 
    shape = 3 X n_features X length '''
    no_stacks = feature.shape[-1]//stack
    feature = feature.reshape(3, -1, feature.shape[2]//3)
    stacked_feature = np.zeros((feature.shape[0], stack*feature.shape[1], feature.shape[-1]//stack))
    for i in range(no_stacks):
        pass
    return feature

def extract_feature(filepath, args):
    samples, sample_rate = sf.read(filepath)
    if args.use_lg_mels == 1:
        S = librosa.feature.melspectrogram(samples.astype(float), sr=sample_rate, n_mels=args.n_mels,\
                                           n_fft=int(sample_rate*(args.n_fft/1000)), \
                                           hop_length=int(sample_rate*(args.hop_length/1000)),\
                                           power=1.0)
        original_len = S.shape[-1]; #print(S.shape)
        if (args.max_frame_len - original_len) >= 0:
            S = np.append(S, (1e-9)*np.ones((args.n_mels, args.max_frame_len-original_len)), axis=1)
            S = librosa.amplitude_to_db(S) # convert to log mel energies
            delta1_S = librosa.feature.delta(S, order=1)
            delta2_S = librosa.feature.delta(S, order=2)
            #S[:,:original_len] = scaler.fit_transform(S[:,:original_len].T).T
            #delta1_S[:,:original_len] = scaler.fit_transform(delta1_S[:,:original_len].T).T
            #delta2_S[:,:original_len] = scaler.fit_transform(delta2_S[:,:original_len].T).T
            features = np.stack((S, delta1_S, delta2_S), axis=0); #break; break;break
            #features = librosa.power_to_db(S, ref=np.max)
            #features = scaler.fit_transform(S.T).T
        else:
            features = None
    else:
        mfcc = librosa.feature.mfcc(y=samples.astype(float), sr=sample_rate, n_mfcc=args.n_mfcc,\
                                                        n_fft=int(sample_rate*(args.n_fft/1000)), \
                                                        hop_length=int(sample_rate*(args.hop_length/1000)))
        original_len = mfcc.shape[-1]
        if (args.max_frame_len - original_len) >= 0:
            mfcc = np.append(mfcc, np.zeros((args.n_mfcc, args.max_frame_len-original_len)), axis=1)
            delta1_mfcc = librosa.feature.delta(mfcc, order=1)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            #mfcc[:,:original_len] = scaler.fit_transform(mfcc[:,:original_len].T).T
            #delta1_mfcc[:,:original_len] = scaler.fit_transform(delta1_mfcc[:,:original_len].T).T
            #delta2_mfcc[:,:original_len] = scaler.fit_transform(delta2_mfcc[:,:original_len].T).T
            features = np.stack((mfcc, delta1_mfcc, delta2_mfcc), axis=0)
        else:
            features = None
    return features, original_len

def get_mfcc_data(args):
    ''' Extracts MFCC 0th, 1st, 2nd order coefficients, tokenizes text transcript, build vocab, convert text tokens to ids
    and saves results in pickle file 
    use_lg_mels: if True, use log of Mel spectrogram instead of MFCC coefficients as features
    '''
    logging.info("Extracting MFCC/Mel features from data...")
    folder = args.folder
    df = pd.DataFrame(columns=['id', 'speaker', 'text','features', 'features_len'])
    tokenizer_en = tokener()
    #scaler = MinMaxScaler(feature_range=(0.0001, 2), copy=True)
    #scaler = StandardScaler()
    def dict_data(data, id_, features=True):
        if id_ in data.keys():
            if features:
                return data[id_][0]
            else:
                return data[id_][1]
    
    for speaker in tqdm(os.listdir("./data/%s/" % folder)):
        for chapter in os.listdir("./data/%s/%s/" % (folder, speaker)):
            data = {}
            for file in os.listdir("./data/%s/%s/%s/" % (folder, speaker, chapter)):
                if '.txt' in file:
                    s_path = "./data/%s/%s/%s/%s" % (folder, speaker, chapter, file)
                    df_dum = pd.read_csv(s_path, names=['id', 'text'])
                    df_dum.loc[:, 'text'] = df_dum.apply(lambda x: separate_info(x['id'])[2], \
                                                          axis=1).apply(lambda x: tokenizer_en.tokenize(x))
                    df_dum.loc[:, 'speaker'] = df_dum.apply(lambda x: separate_info(x['id'])[1], axis=1)
                    df_dum.loc[:, 'id'] = df_dum.apply(lambda x: separate_info(x['id'])[0], axis=1)

                elif '.flac' in file:
                    s_path = "./data/%s/%s/%s/%s" % (folder, speaker, chapter, file)
                    features, original_len = extract_feature(s_path, args)
                    if features is None:
                        continue
                    id_ = re.search("[\d+-]+", file)[0]
                    if (id_ is not None):
                        data[id_] = features, original_len
            
            df_dum["features"] = df_dum.apply(lambda x: dict_data(data, x['id'], features=True), axis=1)
            df_dum["features_len"] = df_dum.apply(lambda x: dict_data(data, x['id'], features=False), axis=1)
            df_dum.dropna(inplace=True)
            df = df.append(df_dum, ignore_index=True)
    
    logger.info("Building vocab and converting transcript to tokens...")
    v = vocab(level=args.level)
    v.build_vocab(df["text"])
    df.loc[:, "text"] = df.apply(lambda x: v.convert_w2idx(x["text"]), axis=1)
    df.loc[:, "text"] = df.apply(lambda x: pad_sos_eos(x["text"]), axis=1)
    
    logger.info("Speaker normalization...")
    logger.info("Getting speakers features mean and std...")
    speaker_stats = {}
    for speaker in tqdm(df["speaker"].unique()):
        df_dum = df[df["speaker"] == speaker]
        channel_mu = []; channel_std = []
        for channel in range(3):
            f_list = []
            for row, l in zip(df_dum["features"], df_dum["features_len"]):
                row = row[:,:,:int(l)]
                f_list.extend(list(row[channel].reshape(-1)))
            f_list = np.array(f_list)
            channel_mu.append(f_list.mean())
            channel_std.append(f_list.std())
        speaker_stats[speaker] = (channel_mu, channel_std)
    save_as_pickle("speaker_stats.pkl", speaker_stats)
    
    logger.info("Normalizing...")
    def speaker_norm(feature, stats):
        channel_mu, channel_std = stats
        for idx, (mu, std) in enumerate(zip(channel_mu, channel_std)):
            feature[idx, :, :] = (feature[idx, :, :] - mu)/std
        return feature
    
    df["features"] = df.apply(lambda x: speaker_norm(x["features"], speaker_stats[x["speaker"]]), axis=1)
    
    logging.info("Saving...")
    save_as_pickle("df_%s.pkl" % folder, df)
    save_as_pickle("vocab.pkl", v)
    logging.info("Saved!")
    
class padded_dataset(Dataset):
    def __init__(self, df, args, labels=True):
        def x_padder(x, max_len):
            if x.shape[-1] < max_len:
                if args.use_lg_mels == 1:
                    x = np.append(x, np.zeros((3, args.n_mels, max_len-x.shape[-1])), axis=2)
                else:
                    x = np.append(x, np.zeros((3, args.n_mfcc, max_len-x.shape[-1])), axis=2)
            return x
        
        def y_padder(x, max_len):
            x = np.array(x, dtype=int)
            if len(x) < max_len:
                x = np.append(x, np.ones((max_len-len(x)), dtype=int)) # 1 is the idx for <pad> token
            return x
        
        self.labels = labels
        self.x_max_len = int(max(df['features_len']))
        self.X = df["features"]
        self.X_len = df["features_len"]
        if labels == True:
            self.y_max_len = max(df['text'].apply(lambda x: len(x)))
            self.y = df["text"].apply(lambda x: y_padder(x, self.y_max_len))
        #self.X = df["features"].apply(lambda x: x_padder(x, self.x_max_len))
        else:
            self.y = y_padder(np.zeros(1, dtype=int), args.max_seq_len)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.labels == True:
            # output X shape = 3 X n_features X length
            return self.X.iloc[idx], self.y.iloc[idx], self.X_len.iloc[idx]
        else:
            return self.X.iloc[idx], self.y, self.X_len.iloc[idx] # return <sos> if labels=False
    
def load_dataloaders(args):
    ''' loads mfcc data, convert words into ids and returns corresponding data_loaders '''
    logger.info("Loading dataloaders...")
    folder = args.folder
    train_path = "./data/" + "df_%s.pkl" % folder
    if not os.path.isfile(train_path):
        get_mfcc_data(args)
    df = load_pickle("df_%s.pkl" % folder)
    
    trainset = padded_dataset(df, args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, \
                              num_workers=0, pin_memory=False)
    train_length = len(trainset)
    max_features_length = trainset.x_max_len
    max_seq_len = trainset.y_max_len
    return train_loader, train_length, max_features_length, max_seq_len
    