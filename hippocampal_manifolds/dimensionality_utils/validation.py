#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:49:34 2022

@author: julio
"""
#VALIDATION OF EMBEDDING. Lightly adapted from UMAP.validation 
#see Venna, Jarkko, and Samuel Kaski.
#"Local multidimensional scaling with controlled tradeoff between 
#trustworthiness and continuity." Proceedings of 5th Workshop on 
#Self-Organizing Maps. 2005. and https://pubmed.ncbi.nlm.nih.gov/16787737/

import numpy as np
import numba
from sklearn.neighbors import KDTree
#from umap.distances import named_distances


@numba.njit(fastmath=True)
def euclidean(x, y):
    """
    Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit()
def trustworthiness_vector_bulk(indices_source, indices_embedded, max_k):

    n_samples = indices_embedded.shape[0]
    trustworthiness = np.zeros(max_k + 1, dtype=np.float64)
    
    for i in range(n_samples):
        for j in range(max_k):
            
            rank = 0
            while indices_source[i, rank] != indices_embedded[i, j]:
                rank += 1

            for k in range(j + 1, max_k + 1):
                if rank > k:
                    trustworthiness[k] += rank - k

    for k in range(1, max_k + 1):
        trustworthiness[k] = 1.0 - trustworthiness[k] * (
            2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
        )
        
    return trustworthiness

@numba.njit()
def continuity_vector_bulk(indices_source, indices_embedded, max_k): 

    n_samples = indices_source.shape[0]
    continuity = np.zeros(max_k + 1, dtype=np.float64)
    
    for i in range(n_samples):
        for j in range(max_k):
            
            rank = 0
            while indices_embedded[i, rank] != indices_source[i, j]:
                rank += 1

            for k in range(j + 1, max_k + 1):
                if rank > k:
                    continuity[k] += rank - k

    for k in range(1, max_k + 1):
        continuity[k] = 1.0 - continuity[k] * (
            2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
        )
        
    return continuity

@numba.njit()
def compute_rank_indices(signal):
    
    n_samples = signal.shape[0]
    dist_vector = np.zeros((n_samples, n_samples))
    indices_signal = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            dist_vector[i,j] = euclidean(signal[i], signal[j])
            dist_vector[j,i] = dist_vector[i,j]
            
    for row in range(indices_signal.shape[0]):
        indices_signal[row] = np.argsort(dist_vector[row])
        
    return indices_signal 


def trustworthiness_vector(source, embedding, max_k, metric="euclidean", indices_source=None):
    tree = KDTree(embedding, metric=metric)
    indices_embedded = tree.query(embedding, k=max_k+1, return_distance=False)
    # Drop the actual point itself
    indices_embedded = indices_embedded[:, 1:]
    if isinstance(indices_source, type(None)):
        #dist = named_distances[metric]
        indices_source = compute_rank_indices(source)
    result = trustworthiness_vector_bulk(indices_source, indices_embedded, max_k)
    
    return result

def continuity_vector(source, embedding, max_k, metric="euclidean", indices_embedded=None): 
    tree = KDTree(source, metric=metric)
    indices_source = tree.query(source, k=max_k+1, return_distance=False)
    # Drop the actual point itself
    indices_source = indices_source[:, 1:]
    if isinstance(indices_embedded, type(None)):
        indices_embedded = compute_rank_indices(embedding)
    result = continuity_vector_bulk(indices_source, indices_embedded, max_k)
    
    return result