# hippocampal_manifolds
This repository contains the code and data associated with the research study: **"Cell-type-specific manifold analysis discloses independent geometric transformations in the hippocampal spatial code."**  


## Overview

In this work, we developed and applied a novel framework for analyzing cell-type-specific manifolds in hippocampal spatial representations. The repository provides the tools and scripts used for:

- Preprocessing and analyzing neural data.
- Computing geometric properties and transformations in neural spaces.
- Simulating hipppocampal low dimensional neural data.
- Applying topological data analysis to high-dimensional neural spaces.

We offer a simple jupyter notebook ([hippocampal_manifolds_example.ipynb](hippocampal_manifolds_example.ipynb)) that goes step by step into some of the analysis. Given the dataset's size (over 20TB) and how expensive some of the analysis are (such as the betti numbers computations), the jupyter notebook only exemplifies some of the analysis. 

<figure>
  <img src="/readme_fig/geometric_transformation_manifold.png" alt="Hippocampal Manifolds" width="1000"/>
  <figcaption>Figure 1: Hippocampal Deep CA1 manifolds while the animal performs a forced-linear-alternation task before and after the maze is rotated 180º within the room. Note how the position encoding  rotates following the maze rotation.</figcaption>
</figure>

## Dataset

The dataset used in this study includes neural recordings from multiple animals across different experimental conditions. 
- Due to its large size (over 20TB), only a subset of animals for each condition is available in [FigShare](https://figshare.com/account/home#/projects/233675).  
- The complete dataset can be requested by contacting us at lmprida@cajal.csic.es

## Code Structure

The code has been structured into different sections:

- **geometric_utils**: a recopilation of the different analysis used to quantify goemetric properties (e.g. ring eccentricity) and transformations (e.g. rotation, displacement) of the low dimensional manifolds.
- **tda_utils**: is composed of different preprocessing steps used to compute topological-data analysis on neural data.
- **dimensionality_utils**: multiple tools used to quantify the intrinsic dimensionality of the neural representation.
- **decoders**: includes different decoder classes as well as functions to decode behavioral functions from neural data and neural manifolds.
- **simulations**: contains the main Class to generate a statistical model that relates single cell tuning properties to neural manifold geometrical characteristics.

## Citation

If you use this code or data in your research, please cite our paper:  

**"Cell-type-specific manifold analysis discloses independent geometric transformations in the hippocampal spatial code."**  
Authors: Julio Esparza, et al.  
Journal: *[Neuron, Volume, Pages, Year]*  
DOI: [Insert DOI]

## License

This project is licensed under the GPL-3.0 License. See the [`LICENSE`](LICENSE) file for details.


## Contact

For questions, data requests, or collaboration inquiries, please reach out to:  

**Liset M. de la Prida**  
Email: **lmprida@cajal.csic.es**  

**Julio Esparza**  
Email: **julioesparzaibanez@gmail.com**  
