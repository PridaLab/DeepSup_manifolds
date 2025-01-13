# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 12:11:52 2022

@author: Usuario
"""
import copy
import numpy as np
import time

#DIM RED LIBRARIES
import umap
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

#METRICS
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import median_absolute_error, f1_score
metrics = {'median_absolute_error': median_absolute_error, 'f1_score': f1_score}

#INNER PACKAGE IMPORTS
from geometric_manifolds.decoders.decoder_classes import DECODERS  #decoders classes

import warnings as warnings
warnings.filterwarnings(action='ignore', category=UserWarning) #supress slice-data warning for XGBoost: 
                                                               #https://stackoverflow.com/questions/67225016/warning-occuring-in-xgboost


def decoders_1D(X = None, Y=None, emb_list = [], nn = None,
                    trial_signal = None, decoder_list = ["wf", "wc", "xgb", "svr"],
                    n_dims = 10, n_splits=10, min_dist = 0.1, metric = 'median_absolute_error', verbose = False):  
    
    """Train decoders on x-base signal (and on projected one if indicated) to 
    predict a 1D y-signal.
    
    Parameters:
    ----------
    X: Numpy array (TXN; rows are time)
        Array containing the base signal in which to train the decoders (rows
        are timestamps, columns are features). It will be also used to compute
        the embeddings and reduced. 
    
    Y: Numpy array (TXN; rows are time) or list (made of those)
        Array or list containing the arrays with the behavioral signal that we
        want the decoders to predict. 
        
    Optional parameters:
    --------------------
    emb_list: List of string
        List containing the name of the embeddings one wants to compute. It
        currently supports: ['pca','isomap','umap'].
        
    trial_signal: Numpy array (TX1; rows are time)
        Array containing the trial each timestamps of the X belongs
        to. It provided, the train/test data set will be computed by dividing
        in half the number of trials (thus, the actual length of the two sets
        will not be identical, but no trial will be splitted in half). If not
        provided the kfold will split in half the data for the train/test set. 
    
    
    decoder_list: List of String (default: ['wf', 'wc', 'xgb', 'svr'])
        List containing the name of the decoders one wants to train/test. It
        currently supports the decoders: ['wf', 'wc', 'xgb', 'svr']
    
    n_dims: Integer (default: 10)
        Number of dimensions to project the data into.
        
    n_splits: Integer (default: 10)
        Number of kfolds to run the training/test. Repeated kfold is used to 
        randomly divide the signal (or trial_signal) into 50%train 50%test to 
        achieve the final number of kfolds.
        
    verbose: Boolean (default: False)
        Boolean specifing whether or not to print verbose relating the progress
        of the training (it mainly prints the kfold it is currently in).
    
    Returns:
    -------
    R2s: Dictionary
        Dictionary containing the training and test median absolute errors for 
        all kfold and all combinations of x_signal/y_signal/decoder.
        
    """
    #ensure Y and emb_list are list
    if isinstance(Y, np.ndarray): Y = list([Y])
    if isinstance(emb_list, str): emb_list = list([emb_list])
    #assert inputs
    assert isinstance(X, np.ndarray), \
        f"'X has to be a numpy array but it was a {type(X)}"
    assert isinstance(Y, list), \
        f"'Y has to be a list of numpy.array but it was a {type(Y)}"
    assert isinstance(emb_list, list), \
        f"'emb_list' has to be a list of string but it was a {type(emb_list)}" 
    #reshape Y from column vectors to 1D-matrix
    for idx, y in enumerate(Y):
        if y.ndim == 1:
            Y[idx] = y.reshape(-1,1)
    #check if trial mat to use when spliting training/test for decoders
    rkf = RepeatedKFold(n_splits=2, n_repeats=np.ceil(n_splits/2).astype(int))
    if isinstance(trial_signal, np.ndarray):
        trial_list = np.unique(trial_signal)
        #train with half of trials
        kfold_signal = trial_list
        total_index = np.linspace(0, X.shape[0]-1, X.shape[0]).astype(int)
    else:
        kfold_signal = X
        
    train_indexes = [];
    test_indexes = [];
    for train_index, test_index in rkf.split(kfold_signal, kfold_signal):
        train_indexes.append(train_index)
        test_indexes.append(test_index)
        
    if verbose:
        print("\t\tKfold: X/X",end='', sep='', flush = True)
        pre_del = '\b\b\b'
    #initialize dictionary to save results
    R2s = dict()
    for emb in ['base_signal', *emb_list]:
        R2s[emb] = dict()
        for decoder_name in decoder_list:
            R2s[emb][decoder_name] = np.zeros((n_splits,len(Y),2))
    
    n_x_signals = len(['base_signal', *emb_list])
    predictions = [[np.zeros((n_splits,y.shape[0],n_x_signals+2)) for x in decoder_list] for y in Y]
    for y_idx, y in enumerate(Y):
        for x_idx in range(len(decoder_list)):
            predictions[y_idx][x_idx][:,:,1] = np.tile(y, (1, n_splits)).T
        
    for kfold_idx in range(n_splits):
        if verbose:
            print(f"{pre_del}{kfold_idx+1}/{n_splits}", sep = '', end='', flush = True)
            pre_del = (len(str(kfold_idx+1))+len(str(n_splits))+1)*'\b'
            
        #split into train and test data
        if isinstance(trial_signal, np.ndarray):
            train_index = np.any(trial_signal.reshape(-1,1)==trial_list[train_indexes[kfold_idx]], axis=1)
            train_index = total_index[train_index]
            test_index = np.any(trial_signal.reshape(-1,1)==trial_list[test_indexes[kfold_idx]], axis=1)
            test_index = total_index[test_index]
        else:
            train_index = train_indexes[kfold_idx]
            test_index = test_indexes[kfold_idx]
            
        for y_idx, y in enumerate(Y):
            for dec_idx in range(len(decoder_list)):
                predictions[y_idx][dec_idx][kfold_idx,train_index,0] = 0
                predictions[y_idx][dec_idx][kfold_idx,test_index,0] = 1

        X_train = []
        X_base_train = X[train_index,:]
        X_train.append(X_base_train)
        X_test = []
        X_base_test = X[test_index,:]
        X_test.append(X_base_test)
        Y_train = [y[train_index,:] for y in Y]
        Y_test = [y[test_index,:] for y in Y]
        #compute embeddings
        for emb in emb_list:
            if 'umap' in emb:
                if isinstance(nn, type(None)):
                    nn = np.round(X_base_train.shape[0]*0.01).astype(int)
                model = umap.UMAP(n_neighbors = nn, n_components =n_dims, min_dist=min_dist)
            elif 'iso' in emb:
                if isinstance(nn, type(None)):
                    nn = np.round(X_base_train.shape[0]*0.01).astype(int)
                model = Isomap(n_neighbors = nn,n_components = n_dims)
            elif 'pca' in emb:
                model = PCA(n_dims)
            X_signal_train = model.fit_transform(X_base_train)
            X_signal_test = model.transform(X_base_test)
            X_train.append(X_signal_train)
            X_test.append(X_signal_test)
        #train and test decoders 
        for emb_idx, emb in enumerate(['base_signal', *emb_list]):
            for y_idx in range(len(Y)):
                for dec_idx, decoder_name in enumerate(decoder_list):
                    #train decoder
                    model_decoder = copy.deepcopy(DECODERS[decoder_name]())
                    model_decoder.fit(X_train[emb_idx], Y_train[y_idx])
                    #make predictions
                    train_pred = model_decoder.predict(X_train[emb_idx])[:,0]
                    test_pred = model_decoder.predict(X_test[emb_idx])[:,0]
                    #check errors
                    if (metric == "f1_score") and (len(np.unique(Y_test[y_idx][:,0])>2)):
                        test_error = metrics[metric](Y_test[y_idx][:,0], test_pred, average="weighted")
                        train_error = metrics[metric](Y_train[y_idx][:,0], train_pred, average="weighted")
                    else:
                        test_error = metrics[metric](Y_test[y_idx][:,0], test_pred)
                        train_error = metrics[metric](Y_train[y_idx][:,0], train_pred)
                    #store results
                    R2s[emb][decoder_name][kfold_idx,y_idx,0] = test_error
                    R2s[emb][decoder_name][kfold_idx,y_idx,1] = train_error

                    total_pred = np.hstack((train_pred, test_pred)).T
                    total_pred = total_pred[np.argsort(np.hstack((train_index, test_index)).T)]
                    predictions[y_idx][dec_idx][kfold_idx,:,emb_idx+2] = total_pred
                    
    if verbose:
        print("", flush = True)            
    return R2s , predictions
