# Deep Learning Small Project

Readme 
How to run 
Setup
Clone the repository. This repository contains the weights, plots and statistics of models 
that we previously ran and train.

How to use this repository
To see the validation set as predicted by the model containing weights that we previously 
trained and finetuned, run the report.ipynb. We also include details about our Dataloader, 
data augmentation techniques, model architecture, finetuning parameters and performance metrics.
To Train the models, run the submit_training.ipynb . Here we run the training models for the two 
classifier and combine the outputs in a 2 stage loader.
We place many of our functions in the functions.py file for modularization.





SUTD [50.039](https://istd.sutd.edu.sg/undergraduate/courses/50-039-theory-and-practice-of-deep-learning) Theory and Practice of Deep Learning - Small Project (9%)

This should work only on Unix-based systems (because of the filepaths)

Installation

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

This repository is collaborated on Google Cloud Platform AI Notebooks with a T4 GPU.

