import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import time
from gensim.models import Word2Vec

from sklearn.metrics import roc_auc_score as auc
# roc_auc_score(y_true, y_pred), y_pred is probability of the greater class
from sklearn.metrics import f1_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

total = pd.read_csv('annotation.csv')
start = dict(zip(total['gene'], total['start']))
base_path = 'drugs/'

def make_feature_list(file, samples):
    """
    list of lists; each sublist - features of a sample.

    file: string, directory of files with features
    samples: list, list of samples to use

    returns:
    features: list of lists
    """
    features = [0] * len(samples)
    # reading
    for i in range(len(samples)):
        # each file is a single line of space-separated features
        with open(file + samples[i] + '_result.tsv', 'r') as f:
            lines = f.readlines()
        if len(lines) > 1:
            print('length error:', len(lines), 'in', samples[i], 'line', i)
        features[i] = lines[0].split()
    #names = [line.split()[0] for line in lines if line.split()[0] in samples]
    return features


def sort_key(ft):
    """
    Key function for sorting along the genome.
    """
    # if agregated: start of gene
    if ft.find('_PF') != -1:
        return start[ft[:ft.find('_PF')]]
    # if broken
    if ft.find('broken') != -1:
        return start[ft[:ft.find('#')]]
    ft = ft.split('#')
    # if only coordinate (no gene)
    if ft[0] == '-':
        return int(ft[1]) # feature is -, coord, description
    # if gene + coord: gene start + coord
    return start[ft[0]] + int(ft[1])



def prep(drug, fld):
    """
    Preparing the data set for Word2Vec training for given fold of given drug.
    """
    # reading test train split
    with open(f'thresholds/{drug}.{fld}_fold_split.txt') as f:
        samples = list(map(lambda x: x[:-1], f.readlines()))
        
    # list of all samples
    with open('all_samples.txt') as f:
        all_samples = set(list(map(lambda x: x[:x.find('_')], f.readlines())))
    
    # objects in train and test sets
    ind = samples.index('test')
    #train_samples = samples[1:ind]
    test_samples = samples[ind+1:]
    
    #test = make_feature_list('feature_lists/', test_samples)
    
    # set for w2v: all except test
    train_w2v_IDs = list(all_samples - set(test_samples))
    train_w2v = make_feature_list('feature_lists/', train_w2v_IDs)
    
    ## add agr features
    with open(f'{base_path}{drug}/agr_feature_list{fld}.txt') as f:
        lines = f.readlines()
    # structure of file:
    # each line is ID then space-separated features
    # set of IDs
    agr = set([line.split()[0] for line in lines])

    for i in range(len(train_w2v)):
        # if there are agr features for this sample, add them
        if train_w2v_IDs[i] in agr:
            train_w2v[i] += lines[ind].split()[1:]
        # sort features
        train_w2v[i].sort(key=sort_key)
    return train_w2v


def train_func_w2v(train_w2v, drug, fld, sizes=[100]):
    """
    Training Word2Vec model.
    """
    win = max([len(l) for l in train_w2v])
    for sz in sizes:
        starttime = time.perf_counter()
        model = Word2Vec(sentences=train_w2v, vector_size=sz, window=win, min_count=1, workers=4, sg=0)
        duration = timedelta(seconds=time.perf_counter()-starttime)
        print(sz, 'took: ', duration)
        model.save(f"word2vec_models/word2vec_{drug}_{fld}_{sz}.model")


def run_w2v(drug, fld):
    train_w2v = prep(drug, fld)
    print(drug, fld)
    train_func_w2v(train_w2v, drug, fld)

    
# here we start lstm training functions
def preproc(model, data):
    """
    Use w2v model to convert data into vectors, skipping features for which encoding doesn't exist.
    """
    possibles = set(model.wv.key_to_index.keys())
    # feature vectors
    tr = [0] * len(data)
    # feature names
    names = [0] * len(data)
    for i in range(len(data)):
        tr[i] = model.wv[[x for x in data[i] if x in possibles]]
        names[i] = [x for x in data[i] if x in possibles]
    return tr, names

    
def prepare_data(drug, fld, return_names=False):
    """
    Making train and test sets.
    """
    # reading test train split
    with open(f'thresholds/{drug}.{fld}_fold_split.txt') as f:
        samples = list(map(lambda x: x[:-1], f.readlines()))
    
    # objects in train and test sets
    ind = samples.index('test')
    train_samples = samples[1:ind]
    test_samples = samples[ind+1:]
    
    # list of lists of features for each sample
    test = make_feature_list('feature_lists/', test_samples)
    train = make_feature_list('feature_lists/', train_samples)

    ## add agr features
    with open(f'{base_path}{drug}/agr_feature_list{fld}.txt') as f:
        lines = f.readlines()
    # structure of file:
    # each line is ID then space-separated features
    # set of IDs
    agr = set([line.split()[0] for line in lines])
    
    # adding to train set
    for i in range(len(train)):
        # if there are agr features for this sample, add them
        if train_samples[i] in agr:
            train[i] += lines[ind].split()[1:]
        # sort features
        train[i].sort(key=sort_key)
    
    # adding to test set
    for i in range(len(test)):
        # if there are agr features for this sample, add them
        if test_samples[i] in agr:
            test[i] += lines[ind].split()[1:]
        # sort features
        test[i].sort(key=sort_key)
    ######

    model = Word2Vec.load(f"word2vec_models/word2vec_{drug}_{fld}_100.model")

    train_vectors, train_names = preproc(model, train)
    test_vectors, test_names = preproc(model, test)

    df = pd.read_csv('all_resistance.csv')
    df = df[['id', drug]].dropna()
    
    y_train = [df.loc[df['id'] == x][drug].iloc[0] for x in train_samples]
    y_test =  [df.loc[df['id'] == x][drug].iloc[0] for x in test_samples]
    print('train', len(y_train))
    print('test', len(y_test))
    train_vectors = [torch.Tensor(x) for x in train_vectors]
    y_train = torch.Tensor(y_train).view(-1, 1).long()

    test_vectors = [torch.Tensor(x) for x in test_vectors]
    y_test = torch.Tensor(y_test).view(-1, 1).long()
    if return_names:
        return train_names, train_vectors, y_train, test_names, test_vectors, y_test
    return train_vectors, y_train, test_vectors, y_test


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output = [batch size, seq_len, hidden_dim]
        attention_scores = self.attn(lstm_output)
        # attention_scores = [batch size, seq_len, 1]
        #attention_scores = attention_scores.squeeze(-1)
        # attention_scores = [batch size, seq_len]
        return F.softmax(attention_scores, dim=-2)


class lstm_attn(nn.Module):
    def __init__(self, drug, fld, input_dim, hidden_dim, bidirectional=False, num_layers=1, dropout=0):
        super().__init__()
        self.drug = drug
        self.fld = fld
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        # input.size(-1) must be equal to input_size.
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional, batch_first=True, num_layers=num_layers, dropout=dropout)
        # attention layer
        self.attn = Attention(hidden_dim + hidden_dim * bidirectional)
        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim + hidden_dim * bidirectional, 2)

    def forward(self, data):
        # outputs all states, (hidden_state, cell_state)
        out, _ = self.lstm(data)
        #print('out:', out.shape)
        # x = hidden[1]  # later update
        weights = self.attn(out)#.unsqueeze(-1)  # adding an extra dimention at the end
        #print(out.shape)
        #print(weights.shape)
        w = out * weights
        #print(w.shape)
        w = w.sum(dim=0)
        x = self.linear(w)
        return x, weights


def train_function(model, loss_fn, optimizer, X_train, y_train, X_test, y_test, m_name, epochs=10, scheduler=None, attn=False, save=False, path='lstm_models/'):
    model.train()
    loss_test = np.zeros(epochs)
    loss_train = np.zeros(epochs)
    f1_test = np.zeros(epochs)
    f1_train = np.zeros(epochs)
    acc_test = np.zeros(epochs)
    acc_train = np.zeros(epochs)
    auc_test = np.zeros(epochs)
    auc_train = np.zeros(epochs)
    recall_test = np.zeros(epochs)
    recall_train = np.zeros(epochs)
    sm = nn.Softmax(dim=-1)
    for ep in range(epochs):
        starttime = time.perf_counter()
        model.train()
        for i in range(len(X_train)):
            #if i > 5:
            #    print('aborting training early: 6 samples')
            #    break
            # if model has attention, it returnes y_pred, attn_weights
            if attn:
                y_pred, _ = model(X_train[i])
            else:
                y_pred = model(X_train[i])
            loss = loss_fn(y_pred.view(1, -1), y_train[i])
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            loss_train[ep] += loss.item()
        if scheduler != None:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            # metrics for test set
            res = [0] * len(X_test)
            for i in range(len(X_test)):
                if attn:
                    res[i], _ = model(X_test[i])
                    #res[i] = res[i].view(1, -1)
                else:
                    res[i] = model(X_test[i]).view(1, -1)
            res = torch.stack(res)
            #print(res.shape)
            f1_test[ep] = f1_score(y_test, res.argmax(dim=-1))
            acc_test[ep] = sum(res.argmax(dim=-1) == y_test[:, 0]).item() / res.shape[0]
            auc_test[ep] = auc(y_test[:, 0], sm(res)[:, 1])
            recall_test[ep] = recall_score(y_test, res.argmax(axis=1))
            l = loss_fn(res, y_test[:, 0])
            loss_test[ep] = l.item()

            # metrics for train set
            res = [0] * len(X_train)
            for i in range(len(X_train)):
                if attn:
                    res[i], _ = model(X_train[i])
                #res[i] = res[i].view(1, -1)
                else:
                    res[i] = model(X_train[i]).view(1, -1)
            res = torch.stack(res)
            f1_train[ep] = f1_score(y_train, res.argmax(dim=-1))
            acc_train[ep] = sum(res.argmax(dim=-1) == y_train[:, 0]).item() / res.shape[0]
            auc_train[ep] = auc(y_train[:, 0], sm(res)[:, 1])
            recall_train[ep] = recall_score(y_train, res.argmax(axis=1))
            l = loss_fn(res, y_train[:, 0])
            loss_train[ep] = l.item()
        if save:
            if not ep % 20 and ep > 0:
                torch.save(model.state_dict(), f'{path}{model.drug}_{model.fld}_training_m{m_name}_ep{ep}.pt')
        duration = timedelta(seconds=time.perf_counter()-starttime)
        print('epoch', ep, 'took: ', duration)
        print('loss train:', f'{loss_train[ep]:.3f}', 'loss test:', f'{loss_test[ep]:.3f}', end=' ')
        print('f1 train:', f'{f1_train[ep]:.3f}', 'f1 test:', f'{f1_test[ep]:.3f}', end=' ')
        print('auc train:', f'{auc_train[ep]:.3f}', 'auc test:', f'{auc_test[ep]:.3f}', end=' ')
        print('recall train:', f'{recall_train[ep]:.3f}', 'recall test:', f'{recall_test[ep]:.3f}')
    metrics = dict()
    metrics['f1_test'] = f1_test
    metrics['f1_train'] = f1_train
    metrics['loss_train'] = loss_train
    metrics['loss_test'] = loss_test
    metrics['acc_test'] = acc_test
    metrics['acc_train'] = acc_train
    metrics['auc_test'] = auc_test
    metrics['auc_train'] = auc_train
    metrics['recall_test'] = recall_test
    metrics['recall_train'] = recall_train
    return metrics


def plot_metrics(metrics, drug, w2v_len, mdl, outfile):
    n = metrics[list(metrics.keys())[0]].shape[0]
    x = np.linspace(1, n, n)
    fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    #ax[0].plot(x, metrics['loss'])
    #ax[0].set_title('Loss')
    mt = ['loss',  'acc', 'recall', 'f1', 'auc']
    for i in range(len(mt)):
        ax[i].plot(x, metrics[mt[i] + '_train'], label='train')
        ax[i].plot(x, metrics[mt[i] + '_test'], label='test')
        ax[i].set_title(mt[i])
        ax[i].legend()
    fig.suptitle(f'{drug}, w2v {w2v_len}, {mdl}')
    plt.tight_layout()
    plt.savefig(outfile, dpi=250)
