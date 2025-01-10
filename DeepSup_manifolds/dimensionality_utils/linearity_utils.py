"""
Created on Wed Dec 11 2024

@author: JulioEI
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.preprocessing import KernelCenterer


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


def participation_ratio(eigenvalues):
    """
    Estimate the number of "dominant" components based on explained variances

    Parameters
    ----------
    eigenvalues : 1D np.ndarray
        explained variance per dimension

    Returns
    -------
    dimensionality estimated using participation ratio formula
    """
    return np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)


def pca_pr(arr):
    """
    Estimate the data's dimensionality using PCA and participation ratio
    
    Parameters
    ----------
    arr : 2D array
        n_samples x n_features data
    
    Returns
    -------
    estimated dimensionality
    """
    pca = PCA().fit(arr)
    return participation_ratio(pca.explained_variance_)


def geodesic_dist_matrix(X, n_neighbors=15, n_jobs=-1):
    """
    Estimate a geodesic distance matrix using nearest neighbors
    
    Parameters
    ----------
    X : 2D array
        n_samples x n_features data
    n_neighbors : int, default 15
        number of nearest neighbors
    n_jobs : int, default -1
        number of cores to use for nearest neighbors calculation
        -1 is use all available
    
    Returns
    -------
    estimated dimensionality
    """
    # borrowed from sklearn.manifold.Isomap to skip the kernel PCA embedding 
    #part because we only need the distance matrix
    nbrs_ = NearestNeighbors(n_neighbors=n_neighbors,
                             algorithm='auto',
                             metric='minkowski',
                             p=2,
                             metric_params=None,
                             n_jobs=n_jobs)
    nbrs_.fit(X)

    kng = kneighbors_graph(nbrs_, n_neighbors,
                           metric='minkowski', p=2,
                           metric_params=None,
                           mode='distance', n_jobs=n_jobs)

    dist_matrix_ = shortest_path(kng,
                            method = 'auto',
                            directed = False)
    
    return dist_matrix_


def geodesic_to_gram_matrix(D):
    """
    Transform the geodesic distance matrix to a Gram matrix used in kernel PCA

    Parameters
    ----------
    D : 2D np.array
        geodesic distance matrix

    Returns
    -------
    K : 2D np.array
    """
    G = D ** 2
    G *= -0.5
    K = KernelCenterer().fit_transform(G)

    return K


def isomap_pr(X, n_neighbors=15, n_jobs=-1):
    """
    Estimate the data's dimensionality using participation ratio and
    the Gram matrix (?) created from the geodesic distance matrix Isomap uses
    
    Parameters
    ----------
    X : 2D array
        n_samples x n_features data
    n_neighbors : int, default 15
        number of nearest neighbors to estimate the geodesic distances
    n_jobs : int, default -1
        number of cores to use for nearest neighbors calculation
        -1 is use all available
    
    Returns
    -------
    estimated dimensionality
    """
    G = geodesic_dist_matrix(X, n_neighbors=n_neighbors, n_jobs=n_jobs)
    K = geodesic_to_gram_matrix(G)
        
    eigvalues = np.real(np.linalg.eigvals(K))
    
    return participation_ratio(eigvalues)
