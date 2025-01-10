# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 12:00:27 2022

@author: Julio
"""


import numpy as np
import random, os, copy
import matplotlib.pyplot as plt
import umap
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_distances
import pickle

class StatisticModel:
    def __init__(self,pos,dir_mov, rot_position= None, rot_dir_movement = None, num_cells=300):
        self.position = pos
        self.num_cells = num_cells
        self.dir_movement = dir_mov

        if not isinstance(rot_position, type(None)):
            self.rot_position = rot_position
        if not isinstance(rot_dir_movement, type(None)):
            self.rot_dir_movement = rot_dir_movement     

        self.min_pos = np.percentile(self.position[:,0],5)
        self.max_pos = np.percentile(self.position[:,0],95)

    def gaus(self,x,sig,x0):
        return np.exp(-(((x-x0)**2)/(2*sig**2)));

    def compute_one_field(self, mu_gauss = 8, sig_gauss = 3, same_prob= 0.2, xmirror_prob=0.3, remap_prob = 0.2, remap_no_field_prob= 0.25, no_field_prob= 0.05):
        min_pos = self.min_pos
        max_pos = self.max_pos
        #stats
        one_field_pd = pd.DataFrame(np.zeros((1,6)),
                            columns=['field_type','main_dir', 'sigma_left','x0_left','sigma_right', 'x0_right'])
            #cell type
        field_type_prob = random.uniform(0,1)
        if field_type_prob<no_field_prob: #no place cell
            field_type = 'no_field'
        elif field_type_prob<no_field_prob+same_prob: #same field on the other direction
            field_type = 'same_field'
        elif field_type_prob<no_field_prob+same_prob+xmirror_prob: #x-mirror
            field_type = 'xmirror_field'
        elif field_type_prob<no_field_prob+same_prob+xmirror_prob+remap_prob:
            field_type = 'remap_field'
        else:
            field_type = 'remap_no_field'
        one_field_pd.loc[0,'field_type'] = field_type
            #main direction
        main_dir_num = int(random.uniform(0,1)<0.5)
        weak_dir_num = np.abs(main_dir_num-1)
        if main_dir_num==0:
            main_dir = 'left'
            weak_dir = 'right'
        else:
            main_dir = 'right'
            weak_dir = 'left'
        one_field_pd.loc[0, 'main_dir'] = main_dir
        main_dir_idx = self.dir_movement==main_dir_num
        weak_dir_idx = self.dir_movement==weak_dir_num
            #sigma
        one_field_pd.loc[0, 'sigma_'+main_dir] = random.gauss(mu_gauss, sig_gauss)
        if (field_type == 'same_field') or (field_type == 'xmirror_field'):
            one_field_pd.loc[0, 'sigma_'+weak_dir] = one_field_pd.loc[0, 'sigma_'+main_dir]
        else:
            one_field_pd.loc[0, 'sigma_'+weak_dir] = random.gauss(mu_gauss, sig_gauss)

            #x0 place field
        if field_type == 'no_field':
            one_field_pd.loc[0, 'x0_'+main_dir] = min_pos-max_pos
            one_field_pd.loc[0, 'x0_'+weak_dir] = min_pos-max_pos
        else:
            one_field_pd.loc[0, 'x0_'+main_dir] = random.uniform(min_pos, max_pos)
            if field_type == 'same_field':
                one_field_pd.loc[0, 'x0_'+weak_dir] = one_field_pd.loc[0, 'x0_'+main_dir]
            elif field_type == 'xmirror_field':
                one_field_pd.loc[0, 'sigma_'+weak_dir] = max_pos-one_field_pd.loc[0, 'x0_'+main_dir]+min_pos
            elif field_type == 'remap_field':
                one_field_pd.loc[0, 'x0_'+weak_dir] = random.uniform(min_pos, max_pos)
            else:
                one_field_pd.loc[0, 'x0_'+weak_dir] = min_pos-max_pos

        return one_field_pd

    def compute_fields(self, mu_gauss= 8, sig_gauss= 3, same_prob= 0.2, xmirror_prob=0.3, remap_prob= 0.2, remap_no_field_prob= 0.25, no_field_prob= 0.05, num_bins = 100):

        assert np.abs(same_prob+xmirror_prob+remap_prob+remap_no_field_prob+no_field_prob-1)<10e-4, \
                f'field probabilities must add 1 currently: {same_prob+xmirror_prob+remap_prob+remap_no_field_prob+no_field_prob}'

        min_pos = self.min_pos
        max_pos = self.max_pos

        self.field_types = pd.DataFrame(np.zeros((self.num_cells,6)),
                                columns=['field_type','main_dir', 'sigma_left','x0_left','sigma_right', 'x0_right'])

        self.field_type_params = {
            'mu_gauss': mu_gauss,
            'sig_gauss': sig_gauss,
            'no_field_prob': no_field_prob,
            'same_prob': same_prob,
            'xmirror_prob': xmirror_prob,
            'remap_prob': remap_prob,
            'remap_no_field_prob': remap_no_field_prob
        }

        self.place_fields = np.zeros((num_bins,self.num_cells,2))
        self.space_bins = np.linspace(min_pos, max_pos, num_bins)

        for cell in range(self.num_cells):

            one_field_pd = self.compute_one_field(mu_gauss, sig_gauss, same_prob, xmirror_prob, remap_prob, remap_no_field_prob, no_field_prob)

            self.field_types.loc[cell,:] = copy.deepcopy(one_field_pd.loc[0,:])
            #compute place_field main direction
            self.place_fields[:, cell, 0] = self.gaus(self.space_bins,self.field_types.loc[cell, 'sigma_right'],
                                                                            self.field_types.loc[cell, 'x0_right'])
            #compute place_field weak direction
            self.place_fields[:, cell, 1] = self.gaus(self.space_bins,self.field_types.loc[cell, 'sigma_left'],
                                                                            self.field_types.loc[cell, 'x0_left'])

    def compute_traces(self,noise_sigma = 0.1):
        if not hasattr(self, 'field_types'):
            self.compute_fields()

        self.signal_params = {
            'noise_sigma': noise_sigma
        }

        self.signal = np.zeros((self.position.shape[0], self.num_cells))
        for cell in range(self.num_cells):

            main_dir = self.field_types.loc[cell,'main_dir']
            if main_dir == 'right': 
                weak_dir = 'left';
                main_dir_num = 0;
                weak_dir_num = 1;

            else: 
                weak_dir='right';
                main_dir_num = 1;
                weak_dir_num = 0;

            main_dir_idx = self.dir_movement==main_dir_num
            weak_dir_idx = self.dir_movement==weak_dir_num

            #compute signal main direction
            self.signal[main_dir_idx, cell] = self.gaus(self.position[main_dir_idx,0],
                                                                self.field_types.loc[cell, 'sigma_'+main_dir],
                                                                self.field_types.loc[cell, 'x0_'+main_dir])
            self.signal[main_dir_idx, cell] += np.random.normal(0, noise_sigma,self.position[main_dir_idx,0].shape[0])

            #compute signal weak direction
            self.signal[weak_dir_idx, cell] = self.gaus(self.position[weak_dir_idx,0],
                                                                self.field_types.loc[cell, 'sigma_'+weak_dir],
                                                                self.field_types.loc[cell, 'x0_'+weak_dir])
            self.signal[weak_dir_idx, cell] += np.random.normal(0, noise_sigma,self.position[weak_dir_idx,0].shape[0])

    def clean_traces(self,smooth_sigma = 5):
        self.signal_params['smooth_sigma'] = smooth_sigma
        if hasattr(self, 'signal'):
            self.clean_signal = gaussian_filter1d(self.signal, sigma = smooth_sigma, axis = 0)

        if hasattr(self, 'rot_signal'):
            self.rot_clean_signal = gaussian_filter1d(self.rot_signal, sigma = smooth_sigma, axis = 0)

    def compute_cell_types(self, allo_prob=0.2, local_anchored_prob=0.6, remap_prob = 0.15, remap_no_field_prob = 0.05):

        assert np.abs(allo_prob+local_anchored_prob+remap_prob+remap_no_field_prob-1)<10e-4, 'cell type probabilities must add 1'

        if not hasattr(self, 'field_types'):
            self.compute_fields()

        min_pos = self.min_pos
        max_pos = self.max_pos

        self.cell_types = pd.DataFrame(np.zeros((self.num_cells,1))*np.nan,
                                columns=['cell_type'])

        self.cell_type_params = {
            'allo_prob': allo_prob,
            'local_anchored_prob': local_anchored_prob,
            'remap_prob': remap_prob,
            'remap_no_field_prob': remap_no_field_prob
        }

        for cell in range(self.num_cells):
            cell_type_prob = random.uniform(0,1)


            old_field_type = self.field_types.loc[cell,'field_type']
            if old_field_type == 'no_field': #it was not a place field before
                if cell_type_prob<remap_prob: #create new place field
                    cell_type = 'remap_cell'
                else: #no field
                    cell_type = 'no_field'
            else:       
                if cell_type_prob<allo_prob: #same field on the other direction
                    cell_type = 'allocentric_cell'
                elif cell_type_prob<allo_prob+local_anchored_prob: #x-mirror
                    cell_type = 'local_anchored_cell'
                elif cell_type_prob<allo_prob+local_anchored_prob+remap_prob:
                    cell_type = 'remap_cell'
                else:
                    cell_type = 'remap_no_field_cell'

            self.cell_types.loc[cell,'cell_type'] = cell_type

    def compute_rotation_fields(self):

        if not hasattr(self, 'cell_types'):
            self.compute_cell_types()

        min_pos = self.min_pos
        max_pos = self.max_pos

        mu_gauss = self.field_type_params['mu_gauss']
        sig_gauss = self.field_type_params['sig_gauss']
        no_field_prob = self.field_type_params['no_field_prob']
        same_prob = self.field_type_params['same_prob']
        xmirror_prob = self.field_type_params['xmirror_prob']
        remap_prob = self.field_type_params['remap_prob']
        remap_no_field_prob = self.field_type_params['remap_no_field_prob']

        self.rotation_field_types = copy.deepcopy(self.field_types)
        self.rotation_place_fields = copy.deepcopy(self.place_fields)*0
        for cell in range(self.num_cells):
            cell_type = self.cell_types.loc[cell,'cell_type']

            if (cell_type == 'no_field') or (cell_type == 'remap_no_field_cell'):
                self.rotation_field_types.loc[cell, 'x0_left'] = min_pos-max_pos
                self.rotation_field_types.loc[cell, 'x0_right'] = min_pos-max_pos
                self.rotation_field_types.loc[cell, 'field_type'] = 'no_field'
                self.rotation_field_types.loc[cell, 'main_dir'] = 'left'


            elif cell_type == 'local_anchored_cell':
                old_main_dir = self.field_types.loc[cell, 'main_dir']
                if old_main_dir == 'right':
                    self.rotation_field_types.loc[cell, 'main_dir'] = 'left'
                else:
                    self.rotation_field_types.loc[cell, 'main_dir'] = 'right'

                old_sigma_left = self.field_types.loc[cell, 'sigma_left']
                old_x0_left = self.field_types.loc[cell, 'x0_left']
                old_sigma_right = self.field_types.loc[cell, 'sigma_right']
                old_x0_right = self.field_types.loc[cell, 'x0_right']

                self.rotation_field_types.loc[cell, 'sigma_left'] = old_sigma_right
                self.rotation_field_types.loc[cell, 'x0_left'] = max_pos-old_x0_right+min_pos
                self.rotation_field_types.loc[cell, 'sigma_right'] = old_sigma_left
                self.rotation_field_types.loc[cell, 'x0_right'] = max_pos-old_x0_left+min_pos

            elif cell_type == 'remap_cell':

                one_field_pd = self.compute_one_field(mu_gauss, sig_gauss, same_prob, xmirror_prob, remap_prob, remap_no_field_prob, no_field_prob)
                self.rotation_field_types.loc[cell,:] = copy.deepcopy(one_field_pd.loc[0,:])


            #compute place_field main direction
            self.rotation_place_fields[:, cell, 0] = self.gaus(self.space_bins,self.rotation_field_types.loc[cell, 'sigma_right'],
                                                                            self.rotation_field_types.loc[cell, 'x0_right'])
            #compute place_field weak direction
            self.rotation_place_fields[:, cell, 1] = self.gaus(self.space_bins,self.rotation_field_types.loc[cell, 'sigma_left'],
                                                                            self.rotation_field_types.loc[cell, 'x0_left'])

    def compute_rotation_traces(self,rot_position=None, rot_dir_movement=None, noise_sigma = None):
        if not hasattr(self, 'rotation_field_types'):
            self.compute_rotation_fields()

        assert not isinstance(rot_position, type(None)) or hasattr(self, 'rot_position'), 'you need to input rot_position'
        assert not isinstance(rot_dir_movement, type(None)) or hasattr(self, 'rot_dir_movement'), 'you need to input rot_position'

        if not isinstance(rot_position, type(None)):
            self.rot_position = rot_position
        if not isinstance(rot_dir_movement, type(None)):
            self.rot_dir_movement = rot_dir_movement     

        if isinstance(noise_sigma, type(None)):
            noise_sigma = self.signal_params['noise_sigma']

        self.rot_signal = np.zeros((self.rot_position.shape[0], self.num_cells))

        for cell in range(self.num_cells):

            main_dir = self.rotation_field_types.loc[cell,'main_dir']
            if main_dir == 'right': 
                weak_dir = 'left';
                main_dir_num = 0;
                weak_dir_num = 1;

            else: 
                weak_dir='right';
                main_dir_num = 1;
                weak_dir_num = 0;

            main_dir_idx = self.rot_dir_movement==main_dir_num
            weak_dir_idx = self.rot_dir_movement==weak_dir_num

            #compute signal main direction
            self.rot_signal[main_dir_idx, cell] = self.gaus(self.rot_position[main_dir_idx,0],
                                                                self.rotation_field_types.loc[cell, 'sigma_'+main_dir],
                                                                self.rotation_field_types.loc[cell, 'x0_'+main_dir])
            self.rot_signal[main_dir_idx, cell] += np.random.normal(0, noise_sigma,self.rot_position[main_dir_idx,0].shape[0])

            #compute signal weak direction
            self.rot_signal[weak_dir_idx, cell] = self.gaus(self.rot_position[weak_dir_idx,0],
                                                                self.rotation_field_types.loc[cell, 'sigma_'+weak_dir],
                                                                self.rotation_field_types.loc[cell, 'x0_'+weak_dir])
            self.rot_signal[weak_dir_idx, cell] += np.random.normal(0, noise_sigma,self.rot_position[weak_dir_idx,0].shape[0])

    def compute_umap_both(self, n_neigh= 120, dim = 3, min_dist = 0.1):

        assert hasattr(self, 'clean_signal'), 'first call self.compute_clean_traces()'
        assert hasattr(self, 'rot_clean_signal'), 'first call self.compute_rotation_clean_traces()'


        signal_both = np.vstack((self.clean_signal, self.rot_clean_signal))
        index_signal = np.zeros((signal_both.shape[0],1))
        index_signal[self.clean_signal.shape[0]:,0] += 1

        self.umap_model = umap.UMAP(n_neighbors=n_neigh, n_components=dim, min_dist=min_dist)
        self.umap_model.fit(signal_both)
        emb_both = self.umap_model.transform(signal_both)

        self.umap_emb_pre = emb_both[index_signal[:,0]==0,:]
        self.umap_emb_rot = emb_both[index_signal[:,0]==1,:]

    def filter_noisy_outliers(self, data, D=None, prctile_th = 10, prctile_noise = 5):
        if isinstance(D, type(None)):
            D = pairwise_distances(data)
        np.fill_diagonal(D, np.nan)
        nn_dist = np.sum(D < np.nanpercentile(D,prctile_th), axis=1)
        noiseIdx = nn_dist < np.percentile(nn_dist, prctile_noise)
        return noiseIdx

    def clean_umap(self, prctile_th = 10, prctile_noise = 5):

        if hasattr(self, 'umap_emb_pre'):
            D = pairwise_distances(self.umap_emb_pre)
            noise_idx = self.filter_noisy_outliers(self.umap_emb_pre,D, prctile_th,prctile_noise)
            self.nout_umap_emb_pre = self.umap_emb_pre[~noise_idx,:]
            self.nout_position = self.position[~noise_idx,:]
            self.nout_dir_movement = self.dir_movement[~noise_idx]

        if hasattr(self, 'umap_emb_rot'):
            D = pairwise_distances(self.umap_emb_rot)
            noise_idx = self.filter_noisy_outliers(self.umap_emb_rot,D, prctile_th,prctile_noise)
            self.nout_umap_emb_rot = self.umap_emb_rot[~noise_idx,:]
            self.nout_rot_position = self.rot_position[~noise_idx,:]
            self.nout_rot_dir_movement = self.rot_dir_movement[~noise_idx]

    def plot_umap(self):

        fig = plt.figure()
        ax = plt.subplot(1,3,1, projection = '3d')
        ax.scatter(*self.nout_umap_emb_pre[:,:3].T, color ='b', s= 20)
        ax.scatter(*self.nout_umap_emb_rot[:,:3].T, color = 'r', s= 20)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax = plt.subplot(1,3,2, projection = '3d')
        ax.scatter(*self.nout_umap_emb_pre[:,:3].T, c = self.nout_position[:,0], s= 20, cmap = 'magma')
        ax.scatter(*self.nout_umap_emb_rot[:,:3].T, c = self.nout_rot_position[:,0], s= 20, cmap = 'magma')
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        dir_color = np.zeros((self.nout_dir_movement.shape[0],3))
        for point in range(self.nout_dir_movement.shape[0]):
            if self.nout_dir_movement[point]==0:
                dir_color[point] = [12/255,136/255,249/255]
            elif self.nout_dir_movement[point]==1:
                dir_color[point] = [17/255,219/255,224/255]
            else:
                dir_color[point] = [14/255,14/255,143/255]

        rot_dir_color = np.zeros((self.nout_rot_dir_movement.shape[0],3))
        for point in range(self.nout_rot_dir_movement.shape[0]):
            if self.nout_rot_dir_movement[point]==0:
                rot_dir_color[point] = [12/255,136/255,249/255]
            elif self.nout_rot_dir_movement[point]==1:
                rot_dir_color[point] = [17/255,219/255,224/255]
            else:
                rot_dir_color[point] = [14/255,14/255,143/255]

        ax = plt.subplot(1,3,3, projection = '3d')
        ax.scatter(*self.nout_umap_emb_pre[:,:3].T, color = dir_color, s= 20)
        ax.scatter(*self.nout_umap_emb_rot[:,:3].T, color = rot_dir_color, s= 20)
        ax.set_xlabel('Dim 1', labelpad = -8)
        ax.set_ylabel('Dim 2', labelpad = -8)
        ax.set_zlabel('Dim 3', labelpad = -8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        return fig

    def get_centroids(self, cloud_A, cloud_B, label_A, label_B, dir_A = None, dir_B = None, num_centroids = 20):
        dims = cloud_A.shape[1]
        if label_A.ndim>1:
            label_A = label_A[:,0]
        if label_B.ndim>1:
            label_B = label_B[:,0]
        #compute label max and min to divide into centroids
        total_label = np.hstack((label_A, label_B))
        label_lims = np.array([(np.percentile(total_label,5), np.percentile(total_label,95))]).T[:,0] 
        #find centroid size
        cent_size = (label_lims[1] - label_lims[0]) / (num_centroids)
        #define centroid edges a snp.ndarray([lower_edge, upper_edge])
        cent_edges = np.column_stack((np.linspace(label_lims[0],label_lims[0]+cent_size*(num_centroids),num_centroids),
                                    np.linspace(label_lims[0],label_lims[0]+cent_size*(num_centroids),num_centroids)+cent_size))


        if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
            cent_A = np.zeros((num_centroids,dims))
            cent_B = np.zeros((num_centroids,dims))
            cent_label = np.mean(cent_edges,axis=1)

            num_centroids_A = np.zeros((num_centroids,))
            num_centroids_B = np.zeros((num_centroids,))
            for c in range(num_centroids):
                points_A = cloud_A[np.logical_and(label_A >= cent_edges[c,0], label_A<cent_edges[c,1]),:]
                cent_A[c,:] = np.median(points_A, axis=0)
                num_centroids_A[c] = points_A.shape[0]
                
                points_B = cloud_B[np.logical_and(label_B >= cent_edges[c,0], label_B<cent_edges[c,1]),:]
                cent_B[c,:] = np.median(points_B, axis=0)
                num_centroids_B[c] = points_B.shape[0]
        else:
            cloud_A_left = copy.deepcopy(cloud_A[dir_A==0,:])
            label_A_left = copy.deepcopy(label_A[dir_A==0])
            cloud_A_right = copy.deepcopy(cloud_A[dir_A==1,:])
            label_A_right = copy.deepcopy(label_A[dir_A==1])
            
            cloud_B_left = copy.deepcopy(cloud_B[dir_B==0,:])
            label_B_left = copy.deepcopy(label_B[dir_B==0])
            cloud_B_right = copy.deepcopy(cloud_B[dir_B==1,:])
            label_B_right = copy.deepcopy(label_B[dir_B==1])
            
            cent_A = np.zeros((2*num_centroids,dims))
            cent_B = np.zeros((2*num_centroids,dims))
            num_centroids_A = np.zeros((2*num_centroids,))
            num_centroids_B = np.zeros((2*num_centroids,))
            
            cent_dir = np.zeros((2*num_centroids, ))
            cent_label = np.tile(np.mean(cent_edges,axis=1),(2,1)).T.reshape(-1,1)
            for c in range(num_centroids):
                points_A_left = cloud_A_left[np.logical_and(label_A_left >= cent_edges[c,0], label_A_left<cent_edges[c,1]),:]
                cent_A[2*c,:] = np.median(points_A_left, axis=0)
                num_centroids_A[2*c] = points_A_left.shape[0]
                points_A_right = cloud_A_right[np.logical_and(label_A_right >= cent_edges[c,0], label_A_right<cent_edges[c,1]),:]
                cent_A[2*c+1,:] = np.median(points_A_right, axis=0)
                num_centroids_A[2*c+1] = points_A_right.shape[0]

                points_B_left = cloud_B_left[np.logical_and(label_B_left >= cent_edges[c,0], label_B_left<cent_edges[c,1]),:]
                cent_B[2*c,:] = np.median(points_B_left, axis=0)
                num_centroids_B[2*c] = points_B_left.shape[0]
                points_B_right = cloud_B_right[np.logical_and(label_B_right >= cent_edges[c,0], label_B_right<cent_edges[c,1]),:]
                cent_B[2*c+1,:] = np.median(points_B_right, axis=0)
                num_centroids_B[2*c+1] = points_B_right.shape[0]

                cent_dir[2*c] = -1
                cent_dir[2*c+1] = 1

        del_cent_nan = np.all(np.isnan(cent_A), axis= 1)+ np.all(np.isnan(cent_B), axis= 1)
        del_cent_num = (num_centroids_A<15) + (num_centroids_B<15)
        del_cent = del_cent_nan + del_cent_num
        
        cent_A = np.delete(cent_A, del_cent, 0)
        cent_B = np.delete(cent_B, del_cent, 0)

        cent_label = np.delete(cent_label, del_cent, 0)

        if isinstance(dir_A, type(None)) or isinstance(dir_B, type(None)):
            return cent_A, cent_B, cent_label
        else:
            cent_dir = np.delete(cent_dir, del_cent, 0)
            return cent_A, cent_B, cent_label, cent_dir

    def find_rotation(self,data_A, data_B, v):
        center_A = data_A.mean(axis=0)
        angles = np.linspace(-np.pi,np.pi,200)
        error = list()
        for angle in angles:
            #https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
            #https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
            a = np.cos(angle/2)
            b = np.sin(angle/2)*v[0]
            c = np.sin(angle/2)*v[1]
            d = np.sin(angle/2)*v[2]
            R = np.array([
                    [a**2+b**2-c**2-d**2, 2*(b*c-a*d), 2*(b*d+a*c)],
                    [2*(b*c+a*d),a**2-b**2+c**2-d**2, 2*(c*d - a*b)],
                    [2*(b*d - a*c), 2*(c*d + a*b), a**2-b**2-c**2+d**2]
                ])

            new_data =np.matmul(R, (data_A-center_A).T).T + center_A
            error.append(np.sum(np.linalg.norm(new_data - data_B, axis=1)))

        return error

    def parametrize_plane(self, point_cloud):
        '''
        point_cloud.shape = [p,d] (points x dimensions)
        based on Ger reply: https://stackoverflow.com/questions/35070178/
        fit-plane-to-a-set-of-points-in-3d-scipy-optimize-minimize-vs-scipy-linalg-lsts
        '''
        #get point cloud center
        cloud_center = point_cloud.mean(axis=0)
        # run SVD
        u, s, vh = np.linalg.svd(point_cloud - cloud_center)
        # unitary normal vector
        norm_vector = vh[2, :]
        return norm_vector, cloud_center

    def check_norm_vector_direction(self, norm_vector, cloud_center, goal_point):
        og_dir = cloud_center+norm_vector
        op_dir = cloud_center+(-1*norm_vector)

        og_distance = np.linalg.norm(og_dir-goal_point)
        op_distance = np.linalg.norm(op_dir-goal_point)
        if og_distance<op_distance:
            return norm_vector
        else:
            return -norm_vector

    def find_rotation_align_vectors(self,a,b):
        v = np.cross(a,b)
        s = np.linalg.norm(v)
        c = np.dot(a,b)

        sscp = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]], [-v[1],v[0],0]])
        sscp2 = np.matmul(sscp,sscp)
        R = np.eye(3)+sscp+(sscp2*(1-c)/s**2)
        return R

    def apply_rotation_to_cloud(self, point_cloud, rotation,center_of_rotation):
        return np.dot(point_cloud-center_of_rotation, rotation) + center_of_rotation
    
    def rotate_cloud_around_axis(self, point_cloud, angle, v):
        cloud_center = point_cloud.mean(axis=0)
        a = np.cos(angle/2)
        b = np.sin(angle/2)*v[0]
        c = np.sin(angle/2)*v[1]
        d = np.sin(angle/2)*v[2]
        R = np.array([
                [a**2+b**2-c**2-d**2, 2*(b*c-a*d), 2*(b*d+a*c)],
                [2*(b*c+a*d),a**2-b**2+c**2-d**2, 2*(c*d - a*b)],
                [2*(b*d - a*c), 2*(c*d + a*b), a**2-b**2-c**2+d**2]
            ])
        return  np.matmul(R, (point_cloud-cloud_center).T).T+cloud_center

    def compute_rotation(self, num_centroids = 40):
        self.cent_pre, self.cent_rot, self.cent_pos, self.cent_dir = self.get_centroids(self.nout_umap_emb_pre, self.nout_umap_emb_rot, 
                                                self.nout_position[:,0], self.nout_rot_position[:,0], 
                                                self.nout_dir_movement, self.nout_rot_dir_movement,
                                                num_centroids=num_centroids)

        #find axis of alignment
        self.norm_vector_pre, self.cloud_center_pre = self.parametrize_plane(self.nout_umap_emb_pre)
        self.norm_vector_rot, self.cloud_center_rot = self.parametrize_plane(self.nout_umap_emb_rot)

        self.norm_vector_pre = self.check_norm_vector_direction(self.norm_vector_pre,self.cloud_center_pre, self.cloud_center_rot)
        self.norm_vector_rot = self.check_norm_vector_direction(self.norm_vector_rot,self.cloud_center_rot, self.cloud_center_pre)

        self.alignment_rot_mat = self.find_rotation_align_vectors(self.norm_vector_pre, -self.norm_vector_rot)  #minus sign to avoid 180 flip
        self.angle_of_alignment = np.arccos(np.clip(np.dot(self.norm_vector_pre, -self.norm_vector_rot), -1.0, 1.0))*180/np.pi

        self.aligned_cent_rot = self.apply_rotation_to_cloud(self.cent_rot, self.alignment_rot_mat, self.cloud_center_rot)
        self.aligned_nout_umap_emb_rot = self.apply_rotation_to_cloud(self.nout_umap_emb_rot, self.alignment_rot_mat, self.cloud_center_rot)
        self.aligned_norm_vector_rot = self.apply_rotation_to_cloud(self.norm_vector_rot, self.alignment_rot_mat, np.array([0,0,0]))

        #project them into planes
        self.plane_nout_umap_emb_pre = self.project_onto_plane(self.nout_umap_emb_pre, self.norm_vector_pre, self.cloud_center_pre)
        self.plane_cent_pre = self.project_onto_plane(self.cent_pre, self.norm_vector_pre, self.cloud_center_pre)

        self.plane_aligned_nout_umap_emb_rot = self.project_onto_plane(self.aligned_nout_umap_emb_rot, self.aligned_norm_vector_rot, self.cloud_center_rot)
        self.plane_aligned_cent_rot = self.project_onto_plane(self.aligned_cent_rot, self.aligned_norm_vector_rot, self.cloud_center_rot)
        self.plane_cent_rot = self.project_onto_plane(self.cent_rot, self.norm_vector_rot, self.cloud_center_rot)
        #compute rotation of planes
        self.angles = np.linspace(-np.pi,np.pi,200)
        self.error = self.find_rotation(self.plane_cent_pre, self.plane_aligned_cent_rot, self.norm_vector_pre)
        self.norm_error = (np.array(self.error)-np.min(self.error))/(np.max(self.error)-np.min(self.error))
        self.signed_rot_angle = self.angles[np.argmin(self.norm_error)]*180/np.pi
        self.rot_angle = np.abs(self.signed_rot_angle)

        #rotate one of the clouds 
        self.rotated_aligned_cent_rot = self.rotate_cloud_around_axis(self.cent_rot, (np.pi/180)*self.signed_rot_angle, self.aligned_norm_vector_rot)
        self.rotated_aligned_nout_umap_emb_rot = self.rotate_cloud_around_axis(self.nout_umap_emb_rot, (np.pi/180)*self.signed_rot_angle, self.aligned_norm_vector_rot)
        self.rotated_plane_aligned_cent_rot = self.rotate_cloud_around_axis(self.plane_aligned_cent_rot, (np.pi/180)*self.signed_rot_angle, self.aligned_norm_vector_rot)
        self.rotated_plane_aligned_nout_umap_emb_rot = self.rotate_cloud_around_axis(self.plane_aligned_nout_umap_emb_rot, (np.pi/180)*self.signed_rot_angle, self.aligned_norm_vector_rot)

    def project_onto_plane(self, point_cloud, norm_plane, point_plane):
        return point_cloud- np.multiply(np.tile(np.dot(norm_plane, (point_cloud-point_plane).T),(3,1)).T,norm_plane)

    def plot_rotation(self):
        def process_axis(ax):
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_zlabel('Dim 3', labelpad = -8)
            ax.set_aspect('equal', adjustable='box')

        scale = self.remap_dist*3
        fig = plt.figure()
        ax = plt.subplot(3,3,1, projection = '3d')
        ax.scatter(*self.nout_umap_emb_pre[:,:3].T, color ='b', s= 10)
        ax.scatter(*self.nout_umap_emb_rot[:,:3].T, color = 'r', s= 10)
        process_axis(ax)

        ax = plt.subplot(3,3,4, projection = '3d')
        ax.scatter(*self.nout_umap_emb_pre[:,:3].T, c = self.nout_dir_movement, s= 10, cmap = 'tab10')
        ax.scatter(*self.nout_umap_emb_rot[:,:3].T, c = self.nout_rot_dir_movement, s= 10, cmap = 'tab10')
        process_axis(ax)

        ax = plt.subplot(3,3,7, projection = '3d')
        ax.scatter(*self.nout_umap_emb_pre[:,:3].T, c = self.nout_position[:,0], s= 10, cmap = 'magma')
        ax.scatter(*self.nout_umap_emb_rot[:,:3].T, c = self.nout_rot_position[:,0], s= 10, cmap = 'magma')
        process_axis(ax)

        ax = plt.subplot(3,3,2, projection = '3d')
        ax.scatter(*self.cent_pre[:,:3].T, color ='b', s= 30)
        ax.scatter(*self.cent_rot[:,:3].T, color = 'r', s= 30)
        ax.scatter(*self.aligned_cent_rot[:,:3].T, color = 'orange', s= 30)
        process_axis(ax)

        ax = plt.subplot(3,3,5, projection = '3d')
        ax.scatter(*self.plane_cent_pre[:,:3].T, c = self.cent_pos[:,0], s= 30, cmap = 'magma')
        ax.scatter(*self.plane_aligned_cent_rot[:,:3].T, c = self.cent_pos[:,0], s= 30, cmap = 'magma')
        ax.plot([self.cloud_center_pre[0], self.cloud_center_pre[0]-scale*self.norm_vector_pre[0]], 
                [self.cloud_center_pre[1], self.cloud_center_pre[1]-scale*self.norm_vector_pre[1]], 
                [self.cloud_center_pre[2], self.cloud_center_pre[2]-scale*self.norm_vector_pre[2]], 
                color = 'b', linewidth = 3)
        ax.plot([self.cloud_center_rot[0], self.cloud_center_rot[0]-scale*self.aligned_norm_vector_rot[0]], 
                [self.cloud_center_rot[1], self.cloud_center_rot[1]-scale*self.aligned_norm_vector_rot[1]], 
                [self.cloud_center_rot[2], self.cloud_center_rot[2]-scale*self.aligned_norm_vector_rot[2]], 
                color = 'r', linewidth = 3)
        for idx in range(self.cent_pos.shape[0]):
            ax.plot([self.plane_cent_pre[idx,0], self.plane_aligned_cent_rot[idx,0]], 
                    [self.plane_cent_pre[idx,1], self.plane_aligned_cent_rot[idx,1]], 
                    [self.plane_cent_pre[idx,2], self.plane_aligned_cent_rot[idx,2]], 
                    color='gray', linewidth=0.5)
        process_axis(ax)

        ax = plt.subplot(3,3,8, projection = '3d')
        ax.scatter(*self.cent_pre[:,:3].T, c = self.cent_pos[:,0], s= 30, cmap = 'magma')
        ax.scatter(*self.rotated_aligned_cent_rot[:,:3].T, c = self.cent_pos[:,0], s= 30, cmap = 'magma')
        ax.plot([self.cloud_center_pre[0], self.cloud_center_pre[0]-scale*self.norm_vector_pre[0]], 
                [self.cloud_center_pre[1], self.cloud_center_pre[1]-scale*self.norm_vector_pre[1]], 
                [self.cloud_center_pre[2], self.cloud_center_pre[2]-scale*self.norm_vector_pre[2]], 
                color = 'b', linewidth = 3)
        ax.plot([self.cloud_center_rot[0], self.cloud_center_rot[0]-scale*self.aligned_norm_vector_rot[0]], 
                [self.cloud_center_rot[1], self.cloud_center_rot[1]-scale*self.aligned_norm_vector_rot[1]], 
                [self.cloud_center_rot[2], self.cloud_center_rot[2]-scale*self.aligned_norm_vector_rot[2]], 
                color = 'r', linewidth = 3)
        for idx in range(self.cent_pos.shape[0]):
            ax.plot([self.plane_cent_pre[idx,0], self.rotated_aligned_cent_rot[idx,0]], 
                    [self.plane_cent_pre[idx,1], self.rotated_aligned_cent_rot[idx,1]], 
                    [self.plane_cent_pre[idx,2], self.rotated_aligned_cent_rot[idx,2]], 
                    color='gray', linewidth=0.5)
        process_axis(ax)


        ax = plt.subplot(1,3,3)
        ax.plot(self.norm_error, self.angles*180/np.pi)
        ax.plot([0,1], [self.signed_rot_angle]*2, '--r')
        ax.set_yticks([-180, -90, 0 , 90, 180])
        ax.set_xlabel('Norm Error')
        ax.set_ylabel('Angle')
        plt.tight_layout()
        plt.suptitle(f"num cells: {self.num_cells} | allo: {self.cell_type_params['allo_prob']:.2f} " +
                    f"| local-anchored: {self.cell_type_params['local_anchored_prob']:.2f} " +
                    f"| remap: {self.cell_type_params['remap_prob']:.2f} " + 
                    f"| no-field: {self.cell_type_params['remap_no_field_prob']:.2f} " + 
                    f" -- rot angle: {self.rot_angle:.2f}")
        return fig

    def fit_ellipse(self, cloud_A, norm_vector):

        rot_2D = self.find_rotation_align_vectors([0,0,1], norm_vector)
        cloud_A_2D = (self.apply_rotation_to_cloud(cloud_A,rot_2D, cloud_A.mean(axis=0)) - cloud_A.mean(axis=0))[:,:2]
        X = cloud_A_2D[:,0:1]
        Y = cloud_A_2D[:,1:]

        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = np.ones_like(X)
        x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

        long_axis = ((x[0]+x[2])/2) + np.sqrt(((x[0]-x[2])/2)**2 + x[1]**2)
        short_axis = ((x[0]+x[2])/2) - np.sqrt(((x[0]-x[2])/2)**2 + x[1]**2)

        x_coord = np.linspace(2*np.min(cloud_A_2D[:,0]),2*np.max(cloud_A_2D[:,0]),1000)
        y_coord = np.linspace(2*np.min(cloud_A_2D[:,1]),2*np.max(cloud_A_2D[:,1]),1000)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord

        a = np.where(np.abs(Z_coord-1)<10e-4)
        fit_ellipse_points = np.zeros((a[0].shape[0],2))
        for point in range(a[0].shape[0]):
            x_coord = X_coord[a[0][point], a[1][point]]
            y_coord = Y_coord[a[0][point], a[1][point]]
            fit_ellipse_points[point,:] = [x_coord, y_coord]

        long_axis = np.max(pairwise_distances(fit_ellipse_points))/2
        short_axis = np.mean(pairwise_distances(fit_ellipse_points))/2


        R_ellipse = self.find_rotation_align_vectors(norm_vector, [0,0,1])

        fit_ellipse_points_3D = np.hstack((fit_ellipse_points, np.zeros((fit_ellipse_points.shape[0],1))))
        fit_ellipse_points_3D = self.apply_rotation_to_cloud(fit_ellipse_points_3D,R_ellipse, fit_ellipse_points_3D.mean(axis=0)) + cloud_A.mean(axis=0)

        return x, long_axis, short_axis, fit_ellipse_points,fit_ellipse_points_3D

    def compute_distance(self):
        if not hasattr(self, 'cent_pre'):
            self.compute_rotation()


        self.inter_dist = np.linalg.norm(self.cloud_center_pre-self.cloud_center_rot)
        self.intra_dist_pre = np.percentile(pairwise_distances(self.cent_pre),95)
        self.intra_dist_rot = np.percentile(pairwise_distances(self.aligned_cent_rot),95)


        self.plane_inter_dist = np.linalg.norm(self.plane_cent_pre.mean(axis=0)-self.plane_cent_rot.mean(axis=0))
        self.ellipse_pre_params, self.ellipse_pre_long_axis, self.ellipse_pre_short_axis, self.ellipse_pre_fit, self.ellipse_pre_fit_3D = self.fit_ellipse(self.plane_cent_pre, self.norm_vector_pre)
        self.ellipse_pre_perimeter = 2*np.pi*np.sqrt(0.5*(self.ellipse_pre_long_axis+self.ellipse_pre_short_axis)**2)

        self.ellipse_rot_params, self.ellipse_rot_long_axis, self.ellipse_rot_short_axis, self.ellipse_rot_fit, self.ellipse_rot_fit_3D = self.fit_ellipse(self.plane_cent_rot, self.norm_vector_rot)
        self.ellipse_rot_perimeter = 2*np.pi*np.sqrt(0.5*(self.ellipse_rot_long_axis+self.ellipse_rot_short_axis)**2)

        self.remap_dist =  self.plane_inter_dist/np.mean((self.ellipse_pre_perimeter, self.ellipse_rot_perimeter))

    def plot_distance(self):

        def process_axis(ax):
            ax.set_xlabel('Dim 1', labelpad = -8)
            ax.set_ylabel('Dim 2', labelpad = -8)
            ax.set_zlabel('Dim 3', labelpad = -8)
            ax.set_aspect('equal', adjustable='box')

        fig = plt.figure(figsize=(14,8))
        ax = plt.subplot(2,3,1, projection = '3d')
        ax.scatter(*self.cent_pre[:,:3].T, color ='b', s= 20)
        ax.scatter(*self.cent_pre.mean(axis=0).T, color ='b', s= 40)
        ax.scatter(*self.cent_rot[:,:3].T, color = 'r', s= 20)
        ax.scatter(*self.cent_rot.mean(axis=0).T, color ='r', s= 40)
        ax.plot([self.cent_pre.mean(axis=0)[0], self.cent_rot.mean(axis=0)[0]],
            [self.cent_pre.mean(axis=0)[1], self.cent_rot.mean(axis=0)[1]],
            [self.cent_pre.mean(axis=0)[2], self.cent_rot.mean(axis=0)[2]],
            color='k', linewidth=2)
        process_axis(ax)

        ax = plt.subplot(2,3,2, projection = '3d')
        ax.scatter(*self.cent_pre[:,:3].T, c = self.cent_dir, s= 20, cmap = 'tab10')
        ax.scatter(*self.cent_rot[:,:3].T, c = self.cent_dir, s= 20, cmap = 'tab10')
        ax.scatter(*self.cent_pre.mean(axis=0).T, color ='b', s= 40)
        ax.scatter(*self.cent_rot.mean(axis=0).T, color ='r', s= 40)
        ax.plot([self.cent_pre.mean(axis=0)[0], self.cent_rot.mean(axis=0)[0]],
            [self.cent_pre.mean(axis=0)[1], self.cent_rot.mean(axis=0)[1]],
            [self.cent_pre.mean(axis=0)[2], self.cent_rot.mean(axis=0)[2]],
            color='k', linewidth=2)
        process_axis(ax)

        ax = plt.subplot(2,3,3, projection = '3d')
        ax.scatter(*self.cent_pre[:,:3].T, c = self.cent_pos[:,0], s= 20, cmap = 'viridis')
        ax.scatter(*self.cent_rot[:,:3].T, c = self.cent_pos[:,0], s= 20, cmap = 'magma')
        ax.scatter(*self.cent_pre.mean(axis=0).T, color ='b', s= 40)
        ax.scatter(*self.cent_rot.mean(axis=0).T, color ='r', s= 40)
        ax.plot([self.cent_pre.mean(axis=0)[0], self.cent_rot.mean(axis=0)[0]],
            [self.cent_pre.mean(axis=0)[1], self.cent_rot.mean(axis=0)[1]],
            [self.cent_pre.mean(axis=0)[2], self.cent_rot.mean(axis=0)[2]],
            color='k', linewidth=2)
        process_axis(ax)

        ax = plt.subplot(2,3,4, projection = '3d')
        ax.scatter(*self.plane_cent_pre[:,:3].T, color ='b', s= 20)
        ax.scatter(*self.plane_cent_pre.mean(axis=0).T, color ='b', s= 40)
        ax.scatter(*self.plane_cent_rot[:,:3].T, color = 'r', s= 20)
        ax.scatter(*self.plane_cent_rot.mean(axis=0).T, color ='r', s= 40)
        ax.plot([self.plane_cent_pre.mean(axis=0)[0], self.plane_cent_rot.mean(axis=0)[0]],
            [self.plane_cent_pre.mean(axis=0)[1], self.plane_cent_rot.mean(axis=0)[1]],
            [self.plane_cent_pre.mean(axis=0)[2], self.plane_cent_rot.mean(axis=0)[2]],
            color='k', linewidth=2)
        ax.scatter(*self.ellipse_pre_fit_3D[:,:3].T, color ='cyan', s= 5,alpha=0.3)
        ax.scatter(*self.ellipse_rot_fit_3D[:,:3].T, color ='orange', s= 5,alpha=0.3)
        process_axis(ax)

        ax = plt.subplot(2,3,5, projection = '3d')
        ax.scatter(*self.plane_cent_pre[:,:3].T, c = self.cent_dir, s= 20, cmap = 'tab10')
        ax.scatter(*self.plane_cent_rot[:,:3].T, c = self.cent_dir, s= 20, cmap = 'tab10')
        ax.scatter(*self.plane_cent_pre.mean(axis=0).T, color ='b', s= 40)
        ax.scatter(*self.plane_cent_rot.mean(axis=0).T, color ='r', s= 40)
        ax.plot([self.plane_cent_pre.mean(axis=0)[0], self.plane_cent_rot.mean(axis=0)[0]],
            [self.plane_cent_pre.mean(axis=0)[1], self.plane_cent_rot.mean(axis=0)[1]],
            [self.plane_cent_pre.mean(axis=0)[2], self.plane_cent_rot.mean(axis=0)[2]],
            color='k', linewidth=2)
        ax.scatter(*self.ellipse_pre_fit_3D[:,:3].T, color ='cyan', s= 5,alpha=0.3)
        ax.scatter(*self.ellipse_rot_fit_3D[:,:3].T, color ='orange', s= 5,alpha=0.3)
        process_axis(ax)

        ax = plt.subplot(2,3,6, projection = '3d')
        ax.scatter(*self.plane_cent_pre[:,:3].T, c = self.cent_pos[:,0], s= 20, cmap = 'viridis')
        ax.scatter(*self.plane_cent_rot[:,:3].T, c = self.cent_pos[:,0], s= 20, cmap = 'magma')
        ax.scatter(*self.plane_cent_pre.mean(axis=0).T, color ='b', s= 40)
        ax.scatter(*self.plane_cent_rot.mean(axis=0).T, color ='r', s= 40)
        ax.plot([self.plane_cent_pre.mean(axis=0)[0], self.plane_cent_rot.mean(axis=0)[0]],
            [self.plane_cent_pre.mean(axis=0)[1], self.plane_cent_rot.mean(axis=0)[1]],
            [self.plane_cent_pre.mean(axis=0)[2], self.plane_cent_rot.mean(axis=0)[2]],
            color='k', linewidth=2)
        ax.scatter(*self.ellipse_pre_fit_3D[:,:3].T, color ='cyan', s= 5,alpha=0.3)
        ax.scatter(*self.ellipse_rot_fit_3D[:,:3].T, color ='orange', s= 5,alpha=0.3)
        process_axis(ax)
        plt.tight_layout()
        return fig