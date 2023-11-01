import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import torchmetrics
import loading_dataset as ld
import CNN_model
import os
from torch.utils.data import DataLoader, random_split
import opendatasets as od


def get_and_split_dataset():
    
    # Loading The Dataset
    full_train_dataset = datasets.ImageFolder(ld.train_data_path,transforms.ToTensor())
    test_dataset = datasets.ImageFolder(ld.test_data_path,transforms.ToTensor())
    
    # Split the datasets into training, validation, and test sets
    train_dataset, valid_dataset, _ = random_split(full_train_dataset, [int(ld.TRAIN_SET_SIZE*0.75), int(ld.TRAIN_SET_SIZE*0.25),0])
    test_dataset, _ , _ = random_split(test_dataset, [ld.TESTING_SET_SIZE, len(test_dataset) - ld.TESTING_SET_SIZE, 0])
    
    return train_dataset,valid_dataset,test_dataset

def tune_mean_std(train_dataset,valid_dataset,test_dataset):

    # Create a DataLoader for the subset
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Calculate mean and standard deviation
    means_stds = []
    loaders = [train_loader,valid_loader,test_loader]
    
    for curr_loader in loaders:
        
        mean = torch.zeros(3)
        std = torch.zeros(3)
        
        for data, _ in curr_loader: 
            
            data = data.view(data.size(0), data.size(1), -1)  # Flatten the images
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            
        mean /= len(curr_loader.dataset)
        std /= len(curr_loader.dataset)
        
        print("Mean:", mean)
        print("Standard Deviation:", std)
        
        means_stds.append([mean,std])

    return means_stds

if __name__ == "__main__":
    
    # train_dataset,valid_dataset,test_dataset = get_and_split_dataset()
    # means_stds = tune_mean_std(train_dataset,valid_dataset,test_dataset)
    