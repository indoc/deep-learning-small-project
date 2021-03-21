# Deep Learning Small Project

SUTD [50.039](https://istd.sutd.edu.sg/undergraduate/courses/50-039-theory-and-practice-of-deep-learning) Theory and Practice of Deep Learning - Small Project (9%)

Our report is at [report.ipynb](./report.ipynb). A [pdf](./report.pdf) version is also available.

This repository contains the weights, plots and statistics of models that we previously ran and train.


### How to use this repository

To see the validation set as predicted by the model containing weights that we previously trained and finetuned, run [report.ipynb](./report.ipynb). 
In this report we also include details about our DataLoader, data augmentation techniques, model architecture, finetuning parameters and performance metrics.

To train the models, run the [submit_training.ipynb](./submit_training.ipynb). Here we run the training models for the two classifier and combine the outputs in a two stage loader.

Our DataLoader is modified from our instructors' sample code, and [lung_dataset.py](./lung_dataset.py) contains our custom DataLoader class - and this script is generated from [lung_dataset.ipynb](./lung_dataset.ipynb)

We place many of our functions in the [functions.py](./functions.py) file for modularization.

This repository will also contain a copy of the datasets, considering that the size of the datasets (27MB) is insignificant compared to the size of the saved models (2.4G).


### Installation


```bash
# install conda
conda create -n dl python=3.7 -y
conda activate dl
conda install scipy numpy matplotlib pytorch h5py scikit-learn -y
conda install jupyter nb_conda ipykernel -y
conda install torchvision -c pytorch -y
```

Additionally for MacOS

```bash
# install brew
brew install tree
```

The code may only work only on Unix-based systems (because of the filepaths).

This repository is collaborated on Google Cloud Platform AI Notebooks with a T4 GPU.

