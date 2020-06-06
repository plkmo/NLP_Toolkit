# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:44:05 2019

@author: WT
"""
import os
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .preprocessing_funcs import load_pickle, save_as_pickle, generate_text_graph
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


def load_datasets(args, train_test_split=0):
    """Loads dataset and graph if exists, else create and process them from raw data
    Returns --->
    f: torch tensor input of GCN (Identity matrix)
    X: input of GCN (Identity matrix)
    A_hat: transformed adjacency matrix A
    selected: indexes of selected labelled nodes for training
    test_idxs: indexes of not-selected nodes for inference/testing
    labels_selected: labels of selected labelled nodes for training
    labels_not_selected: labels of not-selected labelled nodes for inference/testing
    """
    logger.info("Loading data...")
    df_data_path = "./data/df_data.pkl"
    graph_path = "./data/text_graph.pkl"
    if not os.path.isfile(df_data_path) or not os.path.isfile(graph_path):
        logger.info("Building datasets and graph from raw data... Note this will take quite a while...")
        generate_text_graph(args.train_data, args.infer_data, args.max_vocab_len)
    df_data = load_pickle("df_data.pkl")
    G_dict = load_pickle("text_graph.pkl")
    G = G_dict["graph"]
    infer_idx_start = G_dict["infer_idx_start"]
    del G_dict
    
    logger.info("Building adjacency and degree matrices...")
    A = nx.to_numpy_matrix(G, weight="weight"); A = A + np.eye(G.number_of_nodes())
    degrees = []
    for d in G.degree(weight=None):
        if d == 0:
            degrees.append(0)
        else:
            degrees.append(d[1]**(-0.5))
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes()) # Features are just identity matrix
    A_hat = degrees@A@degrees
    f = X # (n X n) X (n X n) x (n X n) X (n X n) input of net
    
    if train_test_split == 1:
        np.random.seed(seed=7)
        logger.info("Splitting labels for training and inferring...")
        ### stratified test samples
        test_idxs = []
        for b_id in df_data["label"].unique():
            dum = df_data[df_data["label"] == b_id]
            if len(dum) >= 4:
                test_idxs.extend(list(np.random.choice(dum.index, size=round(args.test_ratio*len(dum)), replace=False)))
        save_as_pickle("test_idxs.pkl", test_idxs)
        # select only certain labelled nodes for semi-supervised GCN
        selected = []
        for i in range(len(df_data)):
            if i not in test_idxs:
                selected.append(i)
        save_as_pickle("selected.pkl", selected)
    else:
        logger.info("Preparing training labels...")
        test_idxs = [i for i in range(infer_idx_start, len(df_data))]
        selected = [i for i in range(infer_idx_start)]
        save_as_pickle("selected.pkl", selected)
        save_as_pickle("test_idxs.pkl", test_idxs)
    
    f_selected = f[selected]; f_selected = torch.from_numpy(f_selected).float()
    f_not_selected = f[test_idxs]; f_not_selected = torch.from_numpy(f_not_selected).float()
    labels_selected = list(df_data.loc[selected]['label'])
    if train_test_split == 1:    
        labels_not_selected = list(df_data.loc[test_idxs]['label'])
    else:
        labels_not_selected = []
        
    f = torch.from_numpy(f).float()
    save_as_pickle("labels_selected.pkl", labels_selected)
    save_as_pickle("labels_not_selected.pkl", labels_not_selected)
    logger.info("Split into %d train and %d test lebels." % (len(labels_selected), len(labels_not_selected)))
    return f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs
    
def load_state(net, optimizer, scheduler, model_no=0, load_best=False):
    """ Loads saved model and optimizer states if exists """
    logger.info("Initializing model and optimizer states...")
    base_path = "./data/"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % model_no)
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

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./data/test_accuracy_per_epoch_%d.pkl" % model_no
    train_accuracy_path = "./data/train_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path) and os.path.isfile(train_accuracy_path):
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
        train_accuracy_per_epoch = load_pickle("train_accuracy_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, train_accuracy_per_epoch, accuracy_per_epoch = [], [], []
    return losses_per_epoch, train_accuracy_per_epoch, accuracy_per_epoch

def evaluate(output, labels_e):
    if len(labels_e) == 0:
        return 0
    else:
        _, labels = output.max(1); labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
        return sum([(e) for e in labels_e] == labels)/len(labels)

def infer(args, f, test_idxs, net, A_hat, cuda):
    logger.info("Evaluating on inference data...")
    A_hat = torch.FloatTensor(A_hat)
    if cuda:
        A_hat = A_hat.cuda()
    net.eval()
    with torch.no_grad():
        if args.batched == 0:
            pred_labels = net(f)
        elif args.batched == 1:
            pred_labels = net(f, A_hat)
    if pred_labels.is_cuda:
        pred_labels = list(pred_labels[test_idxs].max(1)[1].cpu().numpy())
    else:
        pred_labels = list(pred_labels[test_idxs].max(1)[1].numpy())
    pred_labels = [i for i in pred_labels]
    test_idxs = [i - test_idxs[0] for i in test_idxs]
    df_results = pd.DataFrame(columns=["index", "predicted_label"])
    df_results.loc[:, "index"] = test_idxs
    df_results.loc[:, "predicted_label"] = pred_labels
    df_results.to_csv("./data/results.csv", columns=df_results.columns, index=False)
    logger.info("Done and saved!")
    return df_results

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    T_max : int
        The maximum number of iterations within the first cycle.
    eta_min : float, optional (default: 0)
        The minimum learning rate.
    last_epoch : int, optional (default: -1)
        The index of the last epoch.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs
    
class batched_samples(Dataset):
    def __init__(self, X, A_hat, doc_nodes, args):
        super(batched_samples, self).__init__()
        self.batch_size = args.batch_size
        self.X = X
        self.A_hat = A_hat
        self.nodes = [i for i in range(self.X.shape[0])]
        self.doc_nodes = doc_nodes
        
        self.p_mass = [] # calculate p_mass for importance sampling
        for node in range(self.A_hat.shape[0]):
            p = np.linalg.norm(self.A_hat[:, node])**2
            self.p_mass.append(p)
        self.p_mass = self.p_mass/sum(self.p_mass)
        self.p_mass_doc = self.p_mass[:len(self.doc_nodes)]/sum(self.p_mass[:len(self.doc_nodes)])
        
        assert len(self.nodes) == len(self.p_mass)
        assert round(sum(self.p_mass), 7) == 1.0
        assert len(self.doc_nodes) == len(self.p_mass_doc)
        assert round(sum(self.p_mass_doc), 7) == 1.0
        
    def __len__(self):
        return len(self.doc_nodes)
    
    def nn_sample_nodes(self, idx):
        '''
        NS-type
        '''
        n_nodes = []
        remaining_pool = self.nodes
        another_idx = idx
        while len(n_nodes) < self.batch_size:
            # first get all n-neighbours of node, including itself
            node_idxs = (self.A_hat > 0.0)[another_idx].squeeze().nonzero()[1].tolist()
            if node_idxs is not None:
                n_nodes.extend(node_idxs + [another_idx])
                n_nodes = list(set(n_nodes))
            
            remaining_pool = list(set(remaining_pool).difference(set(n_nodes)))
            another_idx = np.random.choice(remaining_pool, size=1).item()
        return n_nodes
    
    def importance_sample_nodes(self, idx):
        n_doc = np.random.randint(1, int(self.batch_size//2))
        doc_node_ = np.random.choice(self.doc_nodes, size=n_doc, replace=False,\
                                     p=self.p_mass_doc).tolist()
        n_nodes = np.random.choice(self.nodes, size=(self.batch_size - n_doc),\
                                   p=self.p_mass, replace=False).tolist()
        return torch.tensor(list(set(n_nodes + doc_node_)))
    
    def __getitem__(self, idx):
        n_nodes = self.importance_sample_nodes(idx)
        A_batched = torch.FloatTensor(self.A_hat[n_nodes][:, n_nodes])
        X_batched = torch.FloatTensor(self.X[n_nodes])
        return X_batched, A_batched, n_nodes