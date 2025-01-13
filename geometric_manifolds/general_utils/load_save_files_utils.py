"""
Created on Tue Nov 26 2024

@author: JulioEI
"""

import os
import fnmatch
import numpy as np
import pickle
import pandas as pd
import scipy.io


def load_pickle(pickle_path,pickle_name):
    with open(os.path.join(pickle_path, pickle_name), 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(pickle_path, pickle_name, data):
    with open(os.path.join(pickle_path, pickle_name),'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pyaldata_mat_files(data_dir, pattern, verbose=False):
    """
    Load all files in the given directory whose name/type match the pattern specified.
    
    Parameters
    ----------
    data_dir: string with the directory in which to perform the query
    
    pattern: string with the pattern to test the files against before loading 
            (Shortcuts: '*pattern' matches everything, '?pattern' matches any single character, 
             '[seq]pattern' matches any character in seq, '[!seq]pattern' matches any character not in seq)
        
    verbose: False/True
    
    Returns
    -------
    dictionary containing all the files loaded
    """
    fnames = [f for f in os.listdir(data_dir) if fnmatch.fnmatch(os.path.join(data_dir,f), pattern)]
    if verbose:
        print('\tFound ',len(fnames),' files.')
        [print('\t',idx,fname, sep='. ') for idx, fname in enumerate(fnames)];
        print('\tLoading file: 0/', len(fnames), sep = '',end='')
        
    data_dict = dict()
    for idx, file in enumerate(fnames):
        try:
            name_dict = file[:file.find('_PyalData')]
        except:
            name_dict = file[:-3]
        data_dict[name_dict] = mat2dataframe(os.path.join(data_dir,file), shift_idx_fields = True)
        if verbose:
            print('\b\b\b', idx+1, '/', len(fnames), sep = '', end = '')
    if verbose:
        print('')
    if len(data_dict)==1:
        data_dict = data_dict[list(data_dict.keys())[0]]
    return data_dict


def mat2dataframe(path, shift_idx_fields, td_name=None):
    """
    Load a trial_data .mat file and turn it into a pandas DataFrame

    Parameters
    ----------
    path : str
        path to the .mat file to load
        "Can also pass open file-like object."
    td_name : str, optional
        name of the variable under which the data was saved
    shift_idx_fields : bool
        whether to shift the idx fields
        set to True if the data was exported from matlab
        using its 1-based indexig

    Returns
    -------
    df : pd.DataFrame
        pandas dataframe replicating the trial_data format
        each row is a trial
    """
    try:
        mat = scipy.io.loadmat(path, simplify_cells=True)
    except NotImplementedError:
        try:
            import mat73
        except ImportError:
            raise ImportError("Must have mat73 installed to load mat73 files.")
        else:
            mat = mat73.loadmat(path)

    real_keys = [k for k in mat.keys() if not (k.startswith("__") and k.endswith("__"))]

    if td_name is None:
        if len(real_keys) == 0:
            raise ValueError("Could not find dataset name. Please specify td_name.")
        elif len(real_keys) > 1:
            raise ValueError("More than one datasets found. Please specify td_name.")

        assert len(real_keys) == 1

        td_name = real_keys[0]

    df = pd.DataFrame(mat[td_name])

    df = clean_0d_array_fields(df)
    df = clean_integer_fields(df)

    if shift_idx_fields:
        df = backshift_idx_fields(df)

    return df


def backshift_idx_fields(trial_data):
    """
    Adjust index fields from 1-based to 0-based indexing

    Parameters
    ----------
    trial_data : pd.DataFrame
        data in trial_data format

    Returns
    -------
    trial_data with the 'idx' fields adjusted
    """
    idx_fields = [col for col in trial_data.columns.values if "idx" in col]

    for col in idx_fields:
        # using a list comprehension to still work if the idx field itself is an array
        trial_data[col] = [idx - 1 for idx in trial_data[col]]

    return trial_data


def clean_0d_array_fields(df):
    """
    Loading v7.3 MAT files, sometimes scalers are stored as 0-dimensional arrays for some reason.
    This converts those back to scalars.

    Parameters
    ----------
    df : pd.DataFrame
        data in trial_data format

    Returns
    -------
    a copy of df with the relevant fields changed
    """
    for c in df.columns:
        if isinstance(df[c].values[0], np.ndarray):
            if all([arr.ndim == 0 for arr in df[c]]):
                df[c] = [arr.item() for arr in df[c]]

    return df


def clean_integer_fields(df):
    """
    Modify fields that store integers as floats to store them as integers instead.

    Parameters
    ----------
    df : pd.DataFrame
        data in trial_data format

    Returns
    -------
    a copy of df with the relevant fields changed
    """
    for field in df.columns:
        if isinstance(df[field].values[0], np.ndarray):
            try:
                int_arrays = [np.int32(arr) for arr in df[field]]
            except:
                print(f"array field '{field}' could not be converted to int.")
            else:
                if all([np.allclose(int_arr, arr) for (int_arr, arr) in zip(int_arrays, df[field])]):
                    df[field] = int_arrays
        else:
            if not isinstance(df[field].values[0], str):
                try:
                    int_version = np.int32(df[field])
                except:
                        print(f"field '{field}'' could not be converted to int.")
                else:
                    if np.allclose(int_version, df[field]):
                        df[field] = int_version

    return df

