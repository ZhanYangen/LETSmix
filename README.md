Note: This repository is not completely finished yet. Further updates are coming soon.

# SpaDA
[![python >3.8.16](https://img.shields.io/badge/python-3.8.16-brightgreen)](https://www.python.org/) 

This repository provides the official open-source code and public datasets of the paper entitled "a spatially informed and learning-based domain adaptation method for cell-type deconvolution in spatial transcriptomics."

### SpaDA: a spatially informed and learning-based domain adaptation method for cell-type deconvolution in spatial transcriptomics
Spatial transcriptomics (ST) have revolutionized our understanding of gene expression patterns by providing spatial context. However, numerous ST technologies operate on potentially heterogeneous cells mixtures due to the limitation of spatial resolution. Current computational methods designed for cell-type deconvolution often underutilize spatial context information present in ST and the paired high-resolution histopathological images, meanwhile neglect domain variances between ST and the reference single-cell RNA sequencing (scRNA-seq) data. To address these issues, we introduce SpaDA, a novel deep learning-based domain adaptation method trained on labelled pseudo-spots generated from scRNA-seq data, and unlabelled real-spots that are refined by a designed LETS filter leveraging information from spatial correlations among sequenced spots and histological images. The performance of SpaDA is demonstrated across three publicly available ST datasets through comprehensive qualitative and quantitative assessments, setting a new record among current state-of-the-art deconvolution models. Our findings indicate that SpaDA accurately estimates the proportions of various cell types, effectively mapping them to their expected areas within the ST samples. Additionally, the utilization of domain adaptation techniques enables SpaDA to achieve highly stable results when trained with different reference scRNA-seq datasets. Applications of SpaDA to diverse tissues, including the human dorsolateral prefrontal cortex, human pancreatic ductal adenocarcinoma, and mouse liver, showcase its robust performance and generalizability across different biological contexts. 

<img src="model.png" width="800">

# Datasets

All datasets used in the SpaDA paper can be downloaded from the following website: https://zenodo.org/records/10687718. These datasets have been curated and can be directly utilized with codes in this repository. For raw datasets, please refer to the "Data Availability" section in the SpaDA paper.

# Usage

There are three folders under the general "code/" folder, each contains four *.py files named "main", "data", "model", "analysis", respectively, which are codes for reproducing the performance of SpaDA under a specific dataset.
- main.py: 


