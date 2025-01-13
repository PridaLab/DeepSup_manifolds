"""
Created on Mon Dec 30 2024

@author: JulioEI
"""

from sklearn.metrics import pairwise_distances
import numpy as np 
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy
import math 

def filter_noisy_outliers(data, D=None, dist_th = 5, noise_th = 25):
    """
    Filter outliers based on local density. Adapted from Ila 
    Fiete: https://doi.org/10.1038/s41593-019-0460-x

    Parameters
    ----------
    data : numpy array [t,n]
        data  -> t:timestamps; n: neurons
    D: numpy array [t,t]
        matrix of distances between the rows
    
    dist_th: double
        percentile distance threshold below which 
        a point is considered a neighbor

    noise_th: double
        percentile threshold in number of neighbors 
        below which points are considered outliers


    Returns
    -------
    noise_idx: indexes of entries considered outliers
    signal_idx: indexes of entries considered good

    """
    if isinstance(D, type(None)):
        D = pairwise_distances(data)
    np.fill_diagonal(D, np.nan)
    nn_dist = np.sum(D < np.nanpercentile(D,dist_th), axis=1)
    noise_idx = np.where(nn_dist < np.percentile(nn_dist, noise_th))[0]
    signal_idx = np.where(nn_dist >= np.percentile(nn_dist, noise_th))[0]
    return noise_idx, signal_idx

def fuzzy_downsampling(data, num_points=None, num_iters=100, inds=[],  sig=None, w=None, c=None, metric='euclidean'):
    """
    Fuzzy downsampling. Based on Ben Dunn: https://doi.org/10.1038/s41467-024-49703-1
    The function uses exponential decay based on pairwise distances to compute the 
    influence of nearby points.

    Parameters
    ----------
    data : numpy array [t,n]
        data  -> t:timestamps; n: neurons
    num_points: int (<t)
        final number of points to return
    
    Returns
    -------
    data_denoised: numpy array [num_points, n]
    inds: numpy array [num_points,]

    """

    n = np.float64(data.shape[0])
    d = data.shape[1]
    if len(inds)==0:
        inds = np.unique(np.floor(np.arange(0,n-1, n/num_points)).astype(int))
    else:
        num_points = len(inds)
    S = data[inds, :] 
    if not sig:
        sig = np.sqrt(np.var(S))
    if not c:
        c = 0.05*max(pdist(S, metric = metric)) 
    if not w:
        w = 0.3

    dF1 = np.zeros((len(inds), d), float) #gradient of S influenced by all points in data
    dF2 = np.zeros((len(inds), d), float) #gradient of S influenced by S

    for i in range(num_points):
        dF1[i, :] = np.dot((data - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], data, metric = metric), 2) / (2*sig)).T)[:, 0]
        dF2[i, :] = np.dot((S - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], S, metric = metric), 2) / (2*sig)).T)[:, 0]

    dF = 1/sig*(1/n * dF1 - (w / num_points) * dF2) #compute inital final gradient
    M = dF.max()
    for k in range(num_iters):
        S += c*dF/M
        for i in range(num_points):
            dF1[i, :] = np.dot((data - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], data, metric = metric), 2) / (2*sig)).T)[:, 0]
            dF2[i, :] = np.dot((S - S[i, :]).T, np.exp(-np.power(cdist(S[i:i+1, :], S, metric = metric), 2) / (2*sig)).T)[:, 0]
        dF = 1/sig*(1/n * dF1 - (w / num_points) * dF2)
    data_denoised = S
    return data_denoised, inds


def plot_betti_bars(betti, max_dist=1e3, conf_interval = None):
    col_list = ['r', 'g', 'm', 'c']
    betti[0][~np.isfinite(betti[0])] = max_dist
    max_len = [np.max(h) for h in betti]
    max_len = np.max(max_len)
    # Plot the 30 longest barcodes only
    to_plot = []
    for curr_h in betti:
         bar_lens = curr_h[:,1] - curr_h[:,0]
         plot_h = curr_h[(-bar_lens).argsort()[:30]]
         to_plot.append(plot_h[np.argsort(plot_h[:,0]),:])

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(len(betti), 4)
    for curr_betti, curr_bar in enumerate(to_plot):
        ax = fig.add_subplot(gs[curr_betti, :])
        for i, interval in enumerate(reversed(curr_bar)):
            if conf_interval:
                ax.plot([interval[0], interval[0]+conf_interval[curr_betti]], [i,i], linewidth = 5, color =[.8,.8,.8])

            ax.plot([interval[0], interval[1]], [i, i], color=col_list[curr_betti],
                lw=1.5)

        ax.set_ylabel('H' + str(curr_betti))
        ax.set_xlim([-0.1, max_len+0.1])
        # ax.set_xticks([0, xlim])
        ax.set_ylim([-1, len(curr_bar)])
    return fig


def compute_betti(rpp_betti, D):
    betti = list()
    betti.append(np.zeros((rpp_betti[0].shape[0],2)))
    for b in range(betti[0].shape[0]):
        betti[0][b][0] = rpp_betti[0][b][0]
        betti[0][b][1] = rpp_betti[0][b][1]
    betti[0][-1,1] = np.nanmax(D)

    betti.append(np.zeros((rpp_betti[1].shape[0],2)))
    for b in range(betti[1].shape[0]):
        betti[1][b][0] = rpp_betti[1][b][0]
        betti[1][b][1] = rpp_betti[1][b][1]

    betti.append(np.zeros((rpp_betti[2].shape[0],2)))
    for b in range(betti[2].shape[0]):
        betti[2][b][0] = rpp_betti[2][b][0]
        betti[2][b][1] = rpp_betti[2][b][1]
    return betti


def compute_dense_betti(betti, D, dist_to_density = None):

    dense_betti = [np.zeros(x.shape) for x in betti]

    num_edges = int(D.shape[0]*(D.shape[0]-1)/2)
    trilD = D[np.tril_indices_from(D, k=-1)]

    resolution = -int(math.log10(abs(np.max(D)-np.min(D))/1000))

    if isinstance(dist_to_density, type(None)):
        dist_to_density = dict()
    assert isinstance(dist_to_density, dict), "dist_to_density must be dict or None"
    for betti_num in range(len(betti)):
        for bar_num in range(betti[betti_num].shape[0]):
            st = np.round(betti[betti_num][bar_num][0], resolution)
            en = np.round(betti[betti_num][bar_num][1], resolution)
            if st in dist_to_density.keys():
                dense_betti[betti_num][bar_num][0] = dist_to_density[st]
            else:
                dense_st = sum(trilD <= st)/num_edges
                dense_betti[betti_num][bar_num][0] = dense_st
                dist_to_density[st] = dense_st

            if en in dist_to_density.keys():
                dense_betti[betti_num][bar_num][1] = dist_to_density[en]
            else:
                dense_en =  sum(trilD <= en)/num_edges
                dense_betti[betti_num][bar_num][1] = dense_en
                dist_to_density[en] = dense_en

    return dense_betti, dist_to_density





def get_centroids(data, pos, mov_dir = None, num_cent = 20, n_dims = None):
    if n_dims:
        data = data[:,:n_dims]
    else:
        n_dims = data.shape[1]

    if pos.ndim>1:
        pos = pos[:,0]
    #compute label max and min to divide into centroids
    label_limits = np.array([(np.percentile(pos,1), np.percentile(pos,99))]).T[:,0] 
    #find centroid size
    cent_size = (label_limits[1] - label_limits[0]) / (num_cent)
    #define centroid edges a snp.ndarray([lower_edge, upper_edge])
    cent_edges = np.column_stack((np.linspace(label_limits[0],label_limits[0]+cent_size*(num_cent),num_cent),
                                np.linspace(label_limits[0],label_limits[0]+cent_size*(num_cent),num_cent)+cent_size))

    delete_idx = []
    if isinstance(mov_dir, type(None)) :
        cent = np.zeros((num_cent,n_dims))
        num_points_per_cent = np.zeros((num_cent,))
        for c in range(num_cent):
            points = data[np.logical_and(pos >= cent_edges[c,0], pos<cent_edges[c,1]),:]
            if len(points)>0:
                cent[c,:] = np.median(points, axis=0)
                num_points_per_cent[c] = points.shape[0]
            else:
                delete_idx.append(c)
    else:
        data_left = copy.deepcopy(data[mov_dir[:]==-1,:])
        pos_left = copy.deepcopy(pos[mov_dir[:]==-1])
        data_right = copy.deepcopy(data[mov_dir[:]==1,:])
        pos_right = copy.deepcopy(pos[mov_dir[:]==1])
        cent = np.zeros((2*num_cent,n_dims))
        num_points_per_cent = np.zeros((2*num_cent,))
        for c in range(num_cent):
            points_left = data_left[np.logical_and(pos_left >= cent_edges[c,0], pos_left<cent_edges[c,1]),:]
            if len(points_left)>0:
                cent[2*c,:] = np.median(points_left, axis=0)
                num_points_per_cent[2*c] = points_left.shape[0]
            else:
                delete_idx.append(2*c)
            points_right = data_right[np.logical_and(pos_right >= cent_edges[c,0], pos_right<cent_edges[c,1]),:]
            if len(points_right)>0:
                cent[2*c+1,:] = np.median(points_right, axis=0)
                num_points_per_cent[2*c+1] = points_right.shape[0]
            else:
                delete_idx.append(2*c+1)

    # del_cent_nan = np.all(np.isnan(cent), axis= 1)
    # del_cent_num = (num_points_per_cent<20)
    # del_cent = del_cent_nan + del_cent_num
    cent = np.delete(cent, delete_idx, 0)

    return cent, cent_edges