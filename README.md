# MAIS-202-F2023-ASL-Processing

## About the Project

Basic Sign Language Translator that interprets ASL language using Machine Learning from a webcam stream and outputs the corresponding letter:

 * Uses Convolutional Neural Networks
 * Model is trained using [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)
 * Written in Python Version 3, using PyTorch
 
# Requirements
 Use Python Version 3  and install the following packages (tentative list):
* [PyTorch](https://pytorch.org/) and utilities, including torchvision
* [opendatasets](https://pypi.org/project/opendatasets)
* [OpenSSL] version 1.1.1 or higher or [urllib3] to 1.26.7 version pip install urllib3==1.26.7

# Usage
Run [pytorch_version.py](https://github.com/avyan-k/MAIS-202-F2023-ASL-Processing/blob/main/src/pytorch_version.py) to load the data. Training the model will be implemented at a later time.

# Split Dataset
With get_and_split_dataset() we split the dataset into training set (67.5%) validation set (22.5%) and testing set (10%)

# Tuning Weight Regularization
With tune_mean_std() we get the average weight for the mean and standard deviation for the each datasets and we get:

- training dataset:
    - mean:[0.4916, 0.4699, 0.4255]
    - standard deviation:[0.1582, 0.1646, 0.1767]

- validation dataset:
    - mean:[0.4921, 0.4698, 0.4243]
    - standard deviation: [0.1588, 0.1651, 0.1769]

- testing dataset:
    - mean:[0.4904, 0.4686, 0.4246]
    - standard deviation:[0.1587, 0.1656, 0.1776]

We can approximate the best weight by getting the average of weights of all datasets and we get:

- mean: [0.4916,0.4697,0.4251]
- standard deviation: [0.1584,0.1648,0.1768]
