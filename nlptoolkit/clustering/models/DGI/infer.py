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
from sklearn.manifold import TSNE
import torch
from .DGI import DGI
from .train_funcs import load_datasets, X_A_hat, load_state
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
        
        self.df = load_pickle("df_data.pkl")
        self.document_nodes = list(self.df.index)
        
        logger.info("Loading tokenizer and model...")    
        self.G, _ = load_datasets(self.args)
        self.X_A = X_A_hat(self.G)
        X, A_hat = self.X_A.get_X_A_hat(corrupt=False)

        self.net = DGI(X.shape[1], self.args, bias=True,\
                       n_nodes=X.shape[0], A_hat=A_hat, cuda=self.cuda)
        _, _ = load_state(self.net, None, None, model_no=self.args.model_no, load_best=False)
        
        if self.cuda:
            self.net.cuda()
        
        self.net.eval()
        logger.info("Done!")
    
    def infer_embeddings(self, G=None):
        '''
        gets nodes embeddings from trained model
        '''
        self.net.eval()
        
        X, A_hat = self.X_A.get_X_A_hat(corrupt=False)
        if self.cuda:
            X, A_hat = X.cuda(), A_hat.cuda()
        
        with torch.no_grad():
            self.embeddings = self.net.encoder(X, A_hat)
        
        self.embeddings = self.embeddings.cpu().detach().numpy() if self.cuda else\
                            self.embeddings.detach().numpy()
        
        self.embeddings = self.embeddings[self.document_nodes] # only interested in documents
        return self.embeddings
            
    def n_cluster_embeddings(self, features=None, n_clusters=3, method='ac'):
        '''
        clusters the nodes based on embedding features
        features = None (use DGI generated embeddings)
        '''
        if method == 'ac':
            clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',\
                                                 linkage='ward')
            clustering.fit(self.embeddings if features is None else features)
            self.labels = clustering.labels_
            self.score = silhouette_score(self.embeddings if features is None else features,\
                                          self.labels)
        return {'labels': self.labels, 'score': self.score}
    
    def cluster_embeddings(self, features=None, n_start=2, n_stop=5, method='ac', plot=True):
        '''
        vary cluster n size and plot clustering scores
        '''
        self.scores = []
        self.n_range = []
        
        logger.info("Clustering by n...")
        for n in tqdm(range(n_start, n_stop + 1)):
            result = self.n_cluster_embeddings(features=features, n_clusters=n, method=method)
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
    
    def PCA_analyze(self, n_components=2, plot=True):
        '''
        PCA plot
        '''
        pca = PCA(n_components=n_components)
        pca_embeddings = pca.fit_transform(self.embeddings)
        
        if (n_components == 2) and (plot == True):
            fig = plt.figure(figsize=(13,13))
            ax = fig.add_subplot(111)
            ax.scatter(pca_embeddings[:,0], pca_embeddings[:,1], c="red", marker="v", \
                       label="embedded")
            ax.set_xlabel("dim-1", fontsize=15)
            ax.set_ylabel("dim-2", fontsize=15)
            ax.set_title("PCA plot", fontsize=20)
            ax.legend(fontsize=20)
            plt.show()
            plt.close()
        return pca, pca_embeddings
    
    def plot_TSNE(self, plot=True):
        '''
        TSNE plot
        '''
        tsne = TSNE()
        tsne_embeddings = tsne.fit_transform(self.embeddings)
        
        if plot:
            fig = plt.figure(figsize=(13,13))
            ax = fig.add_subplot(111)
            ax.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], c="red", marker="v", \
                       label="embedded")
            ax.set_xlabel("dim-1", fontsize=15)
            ax.set_ylabel("dim-2", fontsize=15)
            ax.set_title("TSNE plot", fontsize=20)
            ax.legend(fontsize=20)
            plt.show()
            plt.close()
        return tsne_embeddings
    
    def cluster_tsne_embeddings(self, tsne_embeddings,\
                                n_start=2, n_stop=30, method='ac', plot=True):
        '''
        Clusters based using TSNE embeddings on DGI node embeddings
        '''
        if n_start != n_stop:
            self.cluster_embeddings(features=tsne_embeddings, n_start=n_start, \
                                    n_stop=n_stop, method=method, plot=plot)
            
            # get best n_cluster
            best_n, best_score = None, -999
            for n, score in zip(self.n_range, self.scores):
                if score > best_score:
                    best_score = score
                    best_n = n
            
            logger.info("Best cluster size: %d" % best_n)
        
        else:
            best_n = n_start
            
        result = self.n_cluster_embeddings(features=tsne_embeddings, \
                                           n_clusters=best_n, method=method)
        
        if plot:
            fig = plt.figure(figsize=(13,13))
            ax = fig.add_subplot(111)
            ax.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], \
                       c=result['labels'], marker="v", \
                       label="embedded labels")
            ax.set_xlabel("dim-1", fontsize=15)
            ax.set_ylabel("dim-2", fontsize=15)
            ax.set_title("TSNE plot (n_cluster = %d)" % best_n, fontsize=20)
            ax.legend(fontsize=20)
            plt.show()
            plt.close()
        return result
    
    def get_clustered_nodes(self, labels):
        '''
        gets node clusters based on their labels
        '''
        node_clusters = {}
        self.df['pred'] = labels
        for label in self.df['pred'].unique():
            node_clusters[label] = self.df[self.df['pred'] == label]
        return node_clusters