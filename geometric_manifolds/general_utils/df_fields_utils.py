import numpy as np
from geometric_manifolds.general_utils import data_process_utils as dp
from geometric_manifolds.general_utils import load_save_files_utils as lsf

import copy 


def get_signal(pd_struct, field_name):
    return copy.deepcopy(np.concatenate(pd_struct[field_name].values, axis=0))

def add_mov_direction_mat_field(pd_struct):

    def compute_movement_direction(position, speed=None, sf = 20, speed_th= 2):
        if isinstance(speed, type(None)):
            speed = np.sqrt(np.sum((np.diff(position, axis=0)*sf)**2,axis=1))
            speed = dp.smooth_data(np.hstack((speed[0], speed)),bin_size=1/sf, std=0.5)

        mov_direction = np.zeros((speed.shape[0],))*np.nan
        mov_direction[speed<speed_th] = 0
        x_speed = np.diff(position[:,0])/(1/sf)
        x_speed = dp.smooth_data(np.hstack((x_speed[0], x_speed)),bin_size=1/sf, std=0.5)
        right_moving = np.logical_and(speed>speed_th, x_speed>0)
        mov_direction[right_moving] = 1
        left_moving = np.logical_and(speed>speed_th, x_speed<0)
        mov_direction[left_moving] = -1
        mov_direction = np.round(dp.smooth_data(mov_direction,bin_size=1/sf, std=0.5),0).astype(int).copy()

        mov_direction_dict = {0: 'non-moving', 1: 'moving to the right', -1: 'moving to the left'}
        return mov_direction, mov_direction_dict
    
    pd_out = copy.deepcopy(pd_struct)
    columns_name = [col for col in pd_out.columns.values]
    lower_columns_name = [col.lower() for col in pd_out.columns.values]

    if 'bin_size' in lower_columns_name:
        sf = 1/pd_out.iloc[0][columns_name[lower_columns_name.index("bin_size")]]
    elif 'fs' in lower_columns_name:
        sf = pd_out.iloc[0][columns_name[lower_columns_name.index("fs")]]
    elif 'sf' in lower_columns_name:
        sf = pd_out.iloc[0][columns_name[lower_columns_name.index("sf")]]

    position = np.concatenate(pd_out["position"].values, axis=0)
    if "speed" in pd_out.columns:
        speed = np.concatenate(pd_out["speed"].values, axis=0)
    else: 
        speed = None

    mov_direction, mov_direction_dict = compute_movement_direction(position, speed, sf)
    if "trial_id_mat" not in lower_columns_name:
        pd_out = add_trial_id_mat_field(pd_out)

    trial_id_mat = np.concatenate(pd_out["trial_id_mat"].values, axis=0).reshape(-1,)

    pd_out["mov_direction"] = [mov_direction[trial_id_mat==pd_out["trial_id"][idx]] 
                                   for idx in pd_out.index]

    pd_out["mov_direction_dict"] = mov_direction_dict         
    return pd_out


def add_trial_id_mat_field(pd_struct):
    pd_out = copy.deepcopy(pd_struct)
    pd_out["trial_id_mat"] = [np.zeros((pd_out["position"][idx].shape[0],))+pd_out["trial_id"][idx] 
                              for idx in pd_out.index]
    return pd_out


def add_trial_type_mat_field(pd_struct):
    pd_out = copy.deepcopy(pd_struct)
    pd_out["trial_type_mat"] = [np.zeros((pd_out["position"][idx].shape[0],)).astype(int)+
                        ('L' == pd_out["dir"][idx])+ 2*('R' == pd_out["dir"][idx])+
                        4*('F' in pd_out["dir"][idx]) for idx in pd_out.index]
    return pd_out


def add_inner_trial_time_field(pd_struct):
    pd_out = copy.deepcopy(pd_struct)
    pd_out["inner_trial_time"] = [np.arange(pd_out["position"][idx].shape[0]).astype(int)
                        for idx in pd_out.index]
    return pd_out


def preprocess_traces_df(pd_struct, field, sig_filt = 5, sig_up = 4, sig_down = 12, peak_th=0.1):
    raw_traces = get_signal(pd_struct, field)
    trial_id_mat = get_signal(pd_struct, 'trial_id_mat')
    clean_traces = dp.preprocess_traces(raw_traces, sig_filt = sig_filt, 
                                            sig_up = sig_up, 
                                            sig_down = sig_down, 
                                            peak_th = peak_th)
    out_pd = copy.deepcopy(pd_struct)
    out_pd['clean_traces'] =  [clean_traces[trial_id_mat==out_pd["trial_id"][idx],:] 
                                          for idx in range(out_pd.shape[0])]
    return out_pd
