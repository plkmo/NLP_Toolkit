#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:48:49 2020

@author: weetee
"""
from nlptoolkit.utils.misc import save_as_pickle
import logging
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default="./data/train.csv", \
                        help="training data csv file path")
    parser.add_argument("--window", type=int, default=10, help='Window size to calculate PMI')
    parser.add_argument("--max_vocab_len", type=int, default=7000, help="GCN encoder: Max vocab size to consider based on top frequency tokens")
    parser.add_argument('--batched', type=int, default=0,\
                        help= 'For GCN, GIN - 0: no batch training ; 1: Yes')
    parser.add_argument("--hidden_size_1", type=int, default=300, help="Size of first GCN encoder hidden weights")
    parser.add_argument("--batch_size", type=int, default=96, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--num_epochs", type=int, default=300, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID: (0: Deep Graph Infomax (DGI)), 
                                                                            ''')
    parser.add_argument("--encoder_type", type=str, default="GIN", \
                        help="For DGI, the encoder type (GCN, GIN)")
    parser.add_argument("--train", type=int, default=1, help="Train model on dataset")
    parser.add_argument("--infer", type=int, default=1, help="Infer input sentence labels from trained model")
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    if args.model_no == 0:
        from nlptoolkit.clustering.models.DGI.trainer import train_and_fit
        from nlptoolkit.clustering.models.DGI.infer import infer_from_trained
    
    if args.train == 1:
        output = train_and_fit(args)
        
    if args.infer == 1:
        inferer = infer_from_trained()
        inferer.infer_embeddings()
        
        pca, pca_embeddings = inferer.PCA_analyze(n_components=2)
        tsne_embeddings = inferer.plot_TSNE(plot=True)
        result = inferer.cluster_tsne_embeddings(tsne_embeddings,\
                                                 n_start=4, n_stop=30, method='ac', plot=True)
        node_clusters = inferer.get_clustered_nodes(result['labels'])
        