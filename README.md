# DeepSup_manifolds
This repository illustrates the data processing and analysis used to study cell-type-specific population representations in the hippocampus, as published in Esparza et al. Neuron 2025 (...). The data can be found in [FigShare](https://figshare.com/account/home#/projects/233675).

We offer a simple jupyter notebook ([DeepSup_manifold_example.ipynb](https://github.com/PridaLab/DeepSup_manifolds/blob/main/DeepSup_manifold_example.ipynb)) that goes step by step into some of the analysis. Given the dataset's size (over 20TB) and how expensive some of the analysis are (such as the betti numbers computations), the jupyter notebook only exemplifies some of the analysis. 

The code has been structured into different sections:


### general_utils
This is composed of general functions that are mostly used to support other processes.

### geometric_utils
A recopilation of the different analysis used to quantify goemetric properties (e.g. ring eccentricity) and transformations (e.g. rotation, displacement) of the low dimensional manifolds.

### tda_utils
A recopilation of different preprocessing steps used to compute topological-data analysis on neural data.

### dimensionality_utils
Multiple tools used to quantify the intrinsic dimensionality of the neural representation.

### decoders
It is includes different decoder classes as well as functions to decode behavioral functions from neural data and neural manifolds.
