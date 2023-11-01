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

## Split Dataset
With get_and_split_dataset() in hyperpara_tuning.py we split the dataset into training set (67.5%) validation set (22.5%) and testing set (10%)

## Tuning Weight Regularization
With tune_mean_std() in hyperpara_tuning.py we get the average weight for the mean and standard deviation for the each datasets. 

After approximating the best weight from it we get:

- mean: [0.4916,0.4697,0.4251]
- standard deviation: [0.1584,0.1648,0.1768]

# Hyperparameters Summary
| Hyperparameters         | Explanation                                              | Computed Result                                     |
|-------------------------|----------------------------------------------------------|-----------------------------------------------------|
| Learning Rate           | Pace the model learns the values of a parameter estimate | {0.01,0.001}                                        |
| Weight Regularization   | Mean and standard deviation for normalizing the dataset  | [[ 0.4916,0.4697,0.4251 ],[ 0.1584,0.1648,0.1768 ]] |
| Convolutional Layers    | Number of convolutional layers in network                |                                                     |
| Initial Output Channels | Number of kernel in the initial convolutional layer      |                                                     |
| Dense Layers            | Number of linear layers after convolution                |                                                     |
| Neurons per Layers      | Number of neurons per dense layers                       |                                                     |
| Dropout                 | Number of neurons we leave out per dense layers          |                                                     |


