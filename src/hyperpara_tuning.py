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
import os
from torch.utils.data import DataLoader, random_split
import opendatasets as od

PATH_TO_DATA = r"synthetic-asl-alphabet"
test_data_path = PATH_TO_DATA + r"/Test_Alphabet"
train_data_path = PATH_TO_DATA + r"/Train_Alphabet"
TRAIN_SET_SIZE = 24300
TESTING_SET_SIZE = 2700

def get_and_split_dataset():
    
    # Loading The Dataset
    full_train_dataset = datasets.ImageFolder(train_data_path,transforms.ToTensor())
    test_dataset = datasets.ImageFolder(test_data_path,transforms.ToTensor())
    
    # Split the datasets into training, validation, and test sets
    train_dataset, valid_dataset, _ = random_split(full_train_dataset, [int(TRAIN_SET_SIZE*0.01), int(TRAIN_SET_SIZE*0.99),0])
    test_dataset, _ , _ = random_split(test_dataset, [TESTING_SET_SIZE, len(test_dataset) - TESTING_SET_SIZE, 0])
    
    return train_dataset,valid_dataset,test_dataset

def find_mean_stds(train_dataset):

    # Create a stack for all images
    train_stack = torch.stack([img_t for img_t, _ in train_dataset], dim=3)
    train_means = train_stack.view(3, -1).mean(dim=1)
    train_stds = train_stack.view(3, -1).std(dim=1)
    return train_means,train_stds


    



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
    train_dataset,valid_dataset,test_dataset = get_and_split_dataset()
    means_stds = find_mean_stds(train_dataset)
    print(means_stds)
    