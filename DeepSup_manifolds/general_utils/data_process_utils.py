# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:24:23 2022

@author: Usuario
"""
import numpy as np
import copy

import scipy.signal as scs
from scipy.ndimage import convolve1d
import warnings
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks

#Adapted from PyalData package (19/10/21) (added assymetry, and variable win_length)
def norm_gauss_window(bin_size, std, num_std = 5, assymetry = False):
    """
    Gaussian window with its mass normalized to 1

    Parameters
    ----------
    bin_size (float): binning size of the array we want to smooth in same time 
                units as the std
    
    std (float): standard deviation of the window use hw_to_std to calculate 
                std based from half-width (same time units as bin_size)
                
    num_std (int): size of the window to convolve in #of stds

    Returns
    -------
    win (1D np.array): Gaussian kernel with length: num_bins*std/bin_length
                mass normalized to 1
    """
    win_len = int(num_std*std/bin_size)
    if win_len%2==0:
        win_len = win_len+1
    win = scs.gaussian(win_len, std/bin_size)
    if assymetry:
        win_2 = scs.gaussian(win_len, 0.5*std/bin_size)
        win[:int((win_len-1)/2)] = win_2[:int((win_len-1)/2)]
        
    return win / np.sum(win)

#Copied from PyalData package (19/10/21)
def hw_to_std(hw):
    """
    Convert half-width to standard deviation for a Gaussian window.
    """
    return hw / (2 * np.sqrt(2 * np.log(2)))

#Copied from PyalData package (19/10/21)
def smooth_data(mat, bin_size=None, std=None, hw=None, win=None, assymetry = False, axis=0):
    """
    Smooth a 1D array or every column of a 2D array

    Parameters
    ----------
    mat : 1D or 2D np.array
        vector or matrix whose columns to smooth
        e.g. recorded spikes in a time x neuron array
    bin_size : float
        length of the timesteps in seconds
    std : float (optional)
        standard deviation of the smoothing window
    hw : float (optional)
        half-width of the smoothing window
    win : 1D array-like (optional)
        smoothing window to convolve with

    Returns
    -------
    np.array of the same size as mat
    """
    #assert mat.ndim == 1 or mat.ndim == 2, "mat has to be a 1D or 2D array"
    assert  sum([arg is not None for arg in [win, hw, std]]) == 1, "only give win, hw or std"
    
    if win is None:
        assert bin_size is not None, "specify bin_size if not supplying window"
        if std is None:
            std = hw_to_std(hw)
        win = norm_gauss_window(bin_size, std, assymetry = assymetry)
    return convolve1d(mat, win, axis=axis, output=np.float32, mode='reflect')

#Copied from PyalData package (19/10/21)
def select_trials(trial_data, query, reset_index=True):
    """
    Select trials based on some criteria

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    query : str, function, array-like
        if array-like, the dataframe is indexed with this
            can be either a list of indices or a mask
        if str, it should express a condition
            it is passed to trial_data.query
        if function/callable, it should take a trial as argument
            and return True for trials you want to keep
    reset_index : bool, optional, default True
        whether to reset the dataframe index to [0,1,2,...]
        or keep the original indices of the kept trials

    Returns
    -------
    trial_data with only the selected trials

    Examples
    --------
    succ_td = select_trials(td, "result == 'R'")

    succ_mask = (td.result == 'R')
    succ_td = select_trials(td, succ_mask)

    train_idx = np.arange(10)
    train_trials = select_trials(td, train_idx)

    right_trials = select_trials(td, lambda trial: np.cos(trial.target_direction) > np.finfo(float).eps)
    """
    if isinstance(query, str):
        trials_to_keep = trial_data.query(query).index
    elif callable(query):
        trials_to_keep = [query(trial) for (i, trial) in trial_data.iterrows()]
    else:
        trials_to_keep = query

    if reset_index:
        return trial_data.loc[trials_to_keep, :].reset_index(drop=True)
    else:
        return trial_data.loc[trials_to_keep, :]
  
def get_neuronal_fields(trial_data, ref_field=None):
    """
    Identify time-varying fields in the dataset
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "spikes" or "rates" is used

    Returns
    -------
    neuronal_fields : list of str
        list of fieldnames that store time-varying signals
    """
    if ref_field is None:
        # look for a spikes field
        ref_field = [col for col in trial_data.columns.values
                     if col.endswith("spikes") or col.endswith("rates") or col.endswith("traces")][0]

    # identify candidates based on the first trial
    first_trial = trial_data.iloc[0]
    T = first_trial[ref_field].shape[1]
    neuronal_fields = []
    for col in first_trial.index:
        try:
            if first_trial[col].shape[1] == T:
                neuronal_fields.append(col)
        except:
            pass

    # but check the rest of the trials, too
    ref_lengths = np.array([arr.shape[1] for arr in trial_data[ref_field]])
    for col in neuronal_fields:
        col_lengths = np.array([arr.shape[1] for arr in trial_data[col]])
        assert np.all(col_lengths == ref_lengths), f"not all lengths in {col} match the reference {ref_field}"

    return neuronal_fields

def get_temporal_fields(trial_data, ref_field=None):
    """
    Identify time-varying fields in the dataset
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    ref_field : str (optional)
        time-varying field to use for identifying the rest
        if not given, the first field that ends with "pos" or "vel" is used

    Returns
    -------
    temporal_fields : list of str
        list of fieldnames that store time-varying signals
    """
    if ref_field is None:
        # look for a spikes field
        ref_field = [col for col in trial_data.columns.values
                     if 'pos' in col or 'traces' in col][0]

    # identify candidates based on the first trial
    first_trial = trial_data.iloc[0]
    T = first_trial[ref_field].shape[0]
    temporal_fields = []
    for col in first_trial.index:
        try:
            if first_trial[col].shape[0] == T:
                temporal_fields.append(col)
        except:
            pass

    # but check the rest of the trials, too
    ref_lengths = np.array([arr.shape[0] for arr in trial_data[ref_field]])
    for col in temporal_fields:
        col_lengths = np.array([arr.shape[0] for arr in trial_data[col]])
        assert np.all(col_lengths == ref_lengths), f"not all lengths in {col} match the reference {ref_field}"

    return temporal_fields

def keep_only_moving(pd_struct, vel_th):
    columns_name = [col for col in pd_struct.columns.values]
    lower_columns_name = [col.lower() for col in pd_struct.columns.values]

    if 'bin_size' in lower_columns_name:
        sf = 1/pd_struct.iloc[0][columns_name[lower_columns_name.index("bin_size")]]
    elif 'fs' in lower_columns_name:
        sf = pd_struct.iloc[0][columns_name[lower_columns_name.index("fs")]]
    elif 'sf' in lower_columns_name:
        sf = pd_struct.iloc[0][columns_name[lower_columns_name.index("sf")]]
    else:
        assert True, "must provide sf"


    if 'trial_id_mat' not in pd_struct:
        pd_struct["trial_id_mat"] = [np.zeros((pd_struct["position"][idx].shape[0],1))+
                                    pd_struct["trial_id"][idx] for idx in pd_struct.index]
    trial_id_mat = copy.deepcopy(np.concatenate(pd_struct['trial_id_mat'].values, axis=0)).reshape(-1,)

    if 'speed' not in pd_struct:
        position = copy.deepcopy(np.concatenate(pd_struct['position'].values, axis=0))
        speed = np.linalg.norm(np.diff(position, axis= 0), axis=1)*sf
        speed = np.hstack((speed[0], speed))
        pd_struct['speed'] = [speed[trial_id_mat==pd_struct["trial_id"][idx]] 
                                                            for idx in pd_struct.index]

    temporal_fields = get_temporal_fields(pd_struct)
    temporal_fields.remove('speed')
    move_struct = copy.deepcopy(pd_struct)
    still_struct = copy.deepcopy(pd_struct)
    for field in temporal_fields:
        move_struct[field] = [pd_struct[field][idx][pd_struct['speed'][idx]>=vel_th]
                                                            for idx in pd_struct.index]
        still_struct[field] = [pd_struct[field][idx][pd_struct['speed'][idx]<vel_th]
                                                            for idx in pd_struct.index]
    move_struct['speed'] = [pd_struct['speed'][idx][pd_struct['speed'][idx]>=vel_th]
                                                            for idx in pd_struct.index]
    still_struct['speed'] = [pd_struct['speed'][idx][pd_struct['speed'][idx]<vel_th]
                                                            for idx in pd_struct.index]
                                                            
    return move_struct, still_struct

def preprocess_traces(traces, sig_filt = 5, sig_up = 4, sig_down = 12, peak_th=0.1):


    lp_traces = uniform_filter1d(traces, size = 4000, axis = 0)
    clean_traces = gaussian_filter1d(traces, sigma = sig_filt, axis = 0)

    for cell in range(clean_traces.shape[1]):
        bleaching = np.histogram(traces[:,cell], 100)
        bleaching = bleaching[1][np.argmax(bleaching[0])]
        bleaching = bleaching + lp_traces[:,cell] - np.min(lp_traces[:,cell]) 

        clean_traces[:,cell] = clean_traces[:,cell]-bleaching
        clean_traces[:,cell] = clean_traces[:,cell]/np.max(clean_traces[:,cell],axis = 0)
    
    clean_traces[clean_traces<0] = 0

    conv_traces = np.zeros(clean_traces.shape)

    gaus = lambda x,sig,amp,vo: amp*np.exp(-(((x)**2)/(2*sig**2)))+vo;
    x = np.arange(-5*sig_down, 5*sig_down,1);
    left_gauss = gaus(x,sig_up, 1, 0); 
    left_gauss[5*sig_down+1:] = 0
    right_gauss = gaus(x,sig_down, 1, 0); 
    right_gauss[:5*sig_down+1] = 0
    gaus_kernel = right_gauss + left_gauss;

    for cell in range(clean_traces.shape[1]):
        peak_idx,_ =find_peaks(clean_traces[:,cell],height=peak_th)
        conv_traces[peak_idx, cell] = clean_traces[peak_idx, cell]
        conv_traces[:, cell] = np.convolve(conv_traces[:, cell],gaus_kernel, 'same')

    return conv_traces



def fix_cross_session_length(df_dict, min_session_len, verbose = False):

    name = ['a','b','c','d','e','f','g']
    #get length of each session
    def recursive_len(item):
        try:
           iter(item)
           return sum(recursive_len(subitem) for subitem in item)
        except TypeError:
           return 1

    session_len = {file: np.sum(pd_struct["position"].apply(recursive_len), axis=0)/2 for file, pd_struct in df_dict.items()}
    bin_size = [pd_struct["bin_size"][0] for _, pd_struct in df_dict.items()]
    if verbose:
        print(f'\tSetting session duration to the shortest one (or to {min_session_len*bin_size[0]:.2f}s): ', sep ='', end= '')

    final_session_len = np.max([min_session_len,  np.min([dur for _, dur in session_len.items()])])
    bin_size = bin_size[np.argmin([dur for _, dur in session_len.items()])]
    if verbose:
        print(f" {int(final_session_len)} samples ({final_session_len*bin_size}s)")

        print('\tOriginal session duration: ')
        [print(f"\t\t{file[:21]}: {int(session_len[file])} samples ({session_len[file]*pd_struct['bin_size'][0]:.2f}s)")
                                                                        for file, pd_struct in df_dict.items()];    
    df_new_dict = dict()
    for file, pd_struct in df_dict.items():

        temporal_fields = get_temporal_fields(pd_struct)
        bin_size = pd_struct["bin_size"][0]
        relative_duration = round(session_len[file]/final_session_len,2)
        
        if relative_duration<0.9: #session last less than 90% of max length
            if verbose:
                print(f"\tSession {file[:21]} last only {100*relative_duration:.2f} of the desired one. Take it into account")
                df_new_dict[file+'_'+name[0]] = pd_struct

        elif 0.9<=relative_duration<=1:
            df_new_dict[file+'_'+name[0]] = pd_struct

        else: 
            num_div = np.ceil(relative_duration).astype(int)
            if verbose:
                print(f"\tSession {file[:21]} lasts {100*relative_duration:.2f} of the desired one. Diving it into {num_div} sections")
            for div in range(num_div-1):
                limit_index = 0
                trial = 0
                consec_length = 0
                stp = 0

                while trial < pd_struct.shape[0] and stp == 0:
                    consec_length += pd_struct["position"][trial].shape[0]
                    if consec_length<final_session_len:
                        trial +=1
                    else:
                        if pd_struct["position"][trial].shape[0]/(consec_length-final_session_len)>0.5:
                            limit_index = trial
                        else:
                            limit_index = trial+1
                        stp = 1
                if stp==1:
                    df_new_dict[file+'_'+name[div]] = copy.deepcopy(pd_struct.iloc[:limit_index,:].reset_index(drop = True))
                    pd_struct = copy.deepcopy(pd_struct.iloc[limit_index+1:, :].reset_index(drop = True))
                else:
                    df_new_dict[file+'_'+name[div]] = pd_struct.reset_index(drop = True)
                    pd_struct = []

            try:    
                new_relative_duration = 0.5*np.sum(pd_struct["position"].apply(recursive_len), axis=0)/final_session_len
            except:
                new_relative_duration = 0
                
            if new_relative_duration<0.8:
                if verbose:
                    print(f"\t\tPart {div+2} lasts only {100*new_relative_duration:.2f} of the desired one. Discarding it")
            elif new_relative_duration<1:
                if verbose:
                    print(f"\t\tPart {div+2} lasts {100*new_relative_duration:.2f} of the desired one. Keeping it")
                df_new_dict[file+'_'+name[div+1]] = copy.deepcopy(pd_struct.reset_index(drop = True))
                
    if verbose:
        new_session_length = {file: np.sum(pd_struct["position"].apply(recursive_len), axis=0)/2 for file, pd_struct in df_new_dict.items()}
        print('\tNew session duration: ')
        [print(f"\t\t{file}: {int(new_session_length[file])} samples ({new_session_length[file]*pd_struct['bin_size'][0]:.2f}s)")
                                                                        for file, pd_struct in df_new_dict.items()];

    return df_new_dict
########################## OLD #########################

#Adapted from PyalData (19/10/21) (add lower case, and continuous option)
def add_firing_rates(data_frame, method, std=None, hw=None, win=None, continuous = False, num_std = 5, assymetry = False):
    """
    Add firing rate fields calculated from spikes fields

    Parameters
    ----------
    trial_data : pd.DataFrame
        trial_data dataframe
    method : str
        'bin' or 'smooth'
    std : float (optional)
        standard deviation of the Gaussian window to smooth with
        default 0.05 seconds
    hw : float (optional)
        half-width of the of the Gaussian window to smooth with
    win : 1D array
        smoothing window

    Returns
    -------
    td : pd.DataFrame
        trial_data with '_rates' fields added
    """
    out_frame = copy.deepcopy(data_frame)
    spike_fields = [name for name in out_frame.columns.values if (name.lower().__contains__("spikes") or name.lower().__contains__("events"))]
    rate_fields = [name.replace("spikes", "rates").replace("events", "revents") for name in spike_fields]
    columns_name = [col for col in out_frame.columns.values]
    lower_columns_name = [col.lower() for col in out_frame.columns.values]
    if 'bin_size' in lower_columns_name:
        bin_size = out_frame.iloc[0][columns_name[lower_columns_name.index("bin_size")]]
    elif 'fs' in lower_columns_name:
        bin_size = 1/out_frame.iloc[0][columns_name[lower_columns_name.index("fs")]]
    elif 'sf' in lower_columns_name:
        bin_size = 1/out_frame.iloc[0][columns_name[lower_columns_name.index("sf")]]
    else:
        raise ValueError('Dataframe does not contain binsize, sf, or fs field.')
        
    assert sum([arg is not None for arg in [win, hw, std]]) == 1, "only give win, hw or std"
    if method == "smooth":
        if win is None:
            if hw is not None:
                std = hw_to_std(hw)
                
            win = norm_gauss_window(bin_size, std, num_std = num_std, assymetry = assymetry)
            
        def get_rate(spikes):
            return smooth_data(spikes, win=win)/bin_size

    elif method == "bin":
        assert all([x is None for x in [std, hw, win]]), "If binning is used, then std, hw, and win have no effect, so don't provide them."
        def get_rate(spikes):
            return spikes/bin_size
    # calculate rates for every spike field
    if not continuous:
        for (spike_field, rate_field) in zip(spike_fields, rate_fields):
            out_frame[rate_field] = [get_rate(spikes) for spikes in out_frame[spike_field]]
    else:
        out_frame["trial_id_mat"] = [np.zeros((out_frame[spike_fields[0]][idx].shape[0],1))+out_frame["trial_id"][idx] 
                                  for idx in range(out_frame.shape[0])]
        trial_id_mat = np.concatenate(out_frame["trial_id_mat"].values, axis=0)
        
        for (spike_field, rate_field) in zip(spike_fields, rate_fields):
            spikes = np.concatenate(out_frame[spike_field], axis = 0)
            rates = get_rate(spikes)
            out_frame[rate_field] = [rates[trial_id_mat[:,0]==out_frame["trial_id"][idx] ,:] 
                                                                for idx in range(out_frame.shape[0])]
    return out_frame



def remove_low_firing_neurons(trial_data, signal, threshold=None, divide_by_bin_size=None, verbose=False, mask= None):
    """
    Remove neurons from signal whose average firing rate
    across all trials is lower than a threshold
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format
    signal : str
        signal from which to calculate the average firing rates
        ideally spikes or rates
    threshold : float
        threshold in Hz
    divide_by_bin_size : bool, optional
        whether to divide by the bin size when calculating the firing rates
    verbose : bool, optional, default False
        print a message about how many neurons were removed

    Returns
    -------
    trial_data with the low-firing neurons removed from the
    signal and the corresponding unit_guide
    """

    if not np.any(mask):
        av_rates = np.mean(np.concatenate(trial_data[signal].values, axis=0), axis=0)
        if divide_by_bin_size:
            av_rates = av_rates/trial_data.bin_size[0]
        mask = av_rates >= threshold
        
    neuronal_fields = get_neuronal_fields(trial_data, ref_field= signal)
    for nfield in neuronal_fields:
        trial_data[nfield] = [arr[:, mask] for arr in trial_data[nfield]]
    
    if signal.endswith("_spikes"):
        suffix = "_spikes"
        unit_guide = signal[:-len(suffix)] + "_unit_guide"

    elif signal.endswith("_rates"):
        suffix = "_rates"
        unit_guide = signal[:-len(suffix)] + "_unit_guide"
    else:
        warnings.warn("Could not determine which unit_guide to modify.")
        unit_guide = None
    if unit_guide in trial_data.columns:
        trial_data[unit_guide] = [arr[mask, :] for arr in trial_data[unit_guide]]
    if verbose:
        print(f"Removed {np.sum(~mask)} neurons from {signal}.")
    return trial_data
