# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:33:49 2019

@author: plkmo
"""
import pickle
import os
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import torch
from .DGI import DGI
from .train_funcs import load_datasets, get_X_A_hat, load_state
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class infer_from_trained(object):
    def __init__(self, args=None):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()
        
        logger.info("Loading tokenizer and model...")    
        self.G = load_datasets(self.args)
        X, A_hat = get_X_A_hat(self.G, corrupt=False)
        #print(labels_selected, labels_not_selected)
        self.net = DGI(X.shape[1], self.args)
        
        _, _ = load_state(self.net, None, None, model_no=self.args.model_no, load_best=False)
        
        if self.cuda:
            self.net.cuda()
        
        self.net.eval()
        logger.info("Done!")
    
    def infer_embeddings(self, G=None):
        self.net.eval()
        if G == None:
            graph = self.G
        else:
            graph = G
        
        X, A_hat = get_X_A_hat(graph, corrupt=False)
        if self.cuda:
            X, A_hat = X.cuda(), A_hat.cuda()
        
        with torch.no_grad():
            self.embeddings = self.net.encoder(X, A_hat)
        
        self.embeddings = self.embeddings.cpu().detach().numpy() if self.cuda else\
                            self.embeddings.detach().numpy()
        
        return self.embeddings
            
    def n_cluster_embeddings(self, n_clusters=3, method='ac'):
        if method == 'ac':
            clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',\
                                                 linkage='ward')
            clustering.fit(self.embeddings)
            self.labels = clustering.labels_
            self.score = silhouette_score(self.embeddings, self.labels)
        return {'labels': self.labels, 'score': self.score}
    
    def cluster_embeddings(self, n_start, n_stop, method='ac', plot=True):
        self.scores = []
        self.n_range = []
        
        logger.info("Clustering by n...")
        for n in tqdm(range(n_start, n_stop + 1)):
            result = self.n_cluster_embeddings(n_clusters=n, method=method)
            self.scores.append(result['score'])
            self.n_range.append(n)
        
        if plot:
            fig = plt.figure(figsize=(13,13))
            ax = fig.add_subplot(111)
            ax.scatter(self.n_range, self.scores, c="red", marker="v", \
                       label="Score")
            ax.set_xlabel("n_cluster", fontsize=15)
            ax.set_ylabel("Score", fontsize=15)
            ax.set_title("%s - Score vs n_cluster" % method, fontsize=20)
            ax.legend(fontsize=20)
            plt.show()
            plt.close()
        return
    
    def PCA_analyze(self, n_components=2):
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(self.embeddings)
        return pca_result