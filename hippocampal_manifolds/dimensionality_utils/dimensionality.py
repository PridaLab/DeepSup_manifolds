"""
Created on Tue Nov 29 2024

@author: JulioEI
"""

from tqdm import tqdm
import numpy as np 
from kneed import KneeLocator
import umap
from hippocampal_manifolds.dimensionality_utils import validation as dimval 
from scipy.spatial import cKDTree


def compute_abids(arr, n_neigh= 50, verbose = True):
    '''
    Compute dimensionality of data array according to Structure Index 
    in UMAP
    
    Parameters:
    -----------
        X (numpy Array): 
            array containing the arr one wants to estimate dimensionality.
        n_neigh (int): 
            number of neighbours used to compute angle.                     
    Returns:
    --------
        (array): 
            array containing the estimated dimensionality for each point in 
            the cloud.
        
    '''
    def abid(X,k,x,search_struct,offset=1):
        neighbor_norms, neighbors = search_struct.query(x,k+offset)
        neighbors = X[neighbors[offset:]] - x
        normed_neighbors = neighbors / neighbor_norms[offset:,None]
        # Original publication version that computes all cosines
        # coss = normed_neighbors.dot(normed_neighbors.T)
        # return np.mean(np.square(coss))**-1
        # Using another product to get the same values with less effort
        para_coss = normed_neighbors.T.dot(normed_neighbors)
        return k**2 / np.sum(np.square(para_coss))

    search_struct = cKDTree(arr)
    if verbose:
        return np.array([
            abid(arr,n_neigh,x,search_struct)
            for x in tqdm(arr,desc="abids",leave=False)
        ])
    else:
        return np.array([abid(arr,n_neigh,x,search_struct) for x in arr])


def compute_umap_dim(X, n_neigh = 5, max_dim = 10):
    '''
    Estimate the data's dimensionality using UMAP trust and cont (see Venna, 
    Jarkko, and Samuel Kaski. Local multidimensional scaling with controlled 
    tradeoff between trustworthiness and continuity." Proceedings of 5th 
    Workshop on Self-Organizing Maps. 2005.)
    
    Parameters
    ----------
    X : 2D array
        n_samples x n_features data
    
    n_neigh: int 
        number of neighbours used to compute trustworthiness.

    max_dim: int
        maximum dimension contemplated

    Returns
    -------
    estimated dimensionality
    '''
    max_dim = np.min([max_dim, X.shape[1]])
    rank_X  = dimval.compute_rank_indices(X)
    trust_num = np.zeros((max_dim,))*np.nan
    cont_num = np.zeros((max_dim,))*np.nan

    for dim in range(np.min([max_dim, X.shape[1]])):
        model = umap.UMAP(n_neighbors = n_neigh, n_components =dim+1)
        emb = model.fit_transform(X)

        #2. Compute trustworthiness
        temp = dimval.trustworthiness_vector(X, emb, n_neigh, 
                                                indices_source = rank_X)
        trust_num[dim] = temp[-1]
        #2. Compute continuity
        temp = dimval.continuity_vector(X, emb, n_neigh)
        cont_num[dim] = temp[-1]

    dim_space = np.arange(1,max_dim+1).astype(int)  
    kl = KneeLocator(dim_space, trust_num, curve = "concave", 
                                                direction = "increasing")
    if kl.knee:
        trust_dim = kl.knee
    else:
        trust_dim = np.nan
    kl = KneeLocator(dim_space, cont_num, curve = "concave", 
                                                direction = "increasing")
    if kl.knee:
        cont_dim = kl.knee
    else:
        cont_dim = np.nan

    hmean_dim = (2*trust_dim*cont_dim)/(trust_dim+cont_dim)
    return trust_num, trust_dim, cont_num, cont_dim, hmean_dim
