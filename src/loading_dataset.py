import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import opendatasets as od
import matplotlib.pyplot as plt
import numpy as np

PATH_TO_DATA = r"synthetic-asl-alphabet"
test_data_path = PATH_TO_DATA + r"/Test_Alphabet"
train_data_path = PATH_TO_DATA + r"/Train_Alphabet"
TRAIN_SET_SIZE=24300
TESTING_SET_SIZE=2700

def download_data():
    
    if not os.path.exists(r"synthetic-asl-alphabet"):
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data")
        
def get_and_split_dataset():
    
    # Loading The Dataset
    full_train_dataset = datasets.ImageFolder(train_data_path,transforms.ToTensor())
    test_dataset = datasets.ImageFolder(test_data_path,transforms.ToTensor())
    
    # Split the datasets into training, validation, and test sets
    train_dataset, valid_dataset, _ = random_split(full_train_dataset, [int(TRAIN_SET_SIZE*0.75), int(TRAIN_SET_SIZE*0.25),0])
    test_dataset, _ , _ = random_split(test_dataset, [TESTING_SET_SIZE, len(test_dataset) - TESTING_SET_SIZE, 0])
    
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
        
def load_data():

    # Data Normalization
    means = [0.4916,0.4697,0.4251]
    standard_deviations = [0.1584,0.1648,0.1768]
    
    # Data Augmentation
    randomizing_transforms = [transforms.RandomRotation(30),transforms.RandomHorizontalFlip()]
   
    # Transform the data to torch tensors and normalize it
    processing_transforms = [transforms.ToTensor(), transforms.Normalize(means, standard_deviations)]
    
    # Loading The Dataset
    full_train_dataset = datasets.ImageFolder(train_data_path, transform=transforms.Compose(randomizing_transforms + processing_transforms))
    test_dataset = datasets.ImageFolder(test_data_path, transform=transforms.Compose(processing_transforms))
    
    # Split the datasets into training, validation, and test sets
    train_dataset, valid_dataset, _ = torch.utils.data.random_split(full_train_dataset, [int(TRAIN_SET_SIZE*0.75), int(TRAIN_SET_SIZE*0.25),0])
    test_dataset, _, _ = torch.utils.data.random_split(test_dataset, [TESTING_SET_SIZE, len(test_dataset) - TESTING_SET_SIZE, 0])
    
    # Set Batch Size and shuffles the data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=100, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader

def load_device():
    
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available(): # If a GPU is available, use it
        device = torch.device("cuda")
    else: # Else, revert to the default (CPU)
        device = torch.device("cpu")
    return device

if __name__ == "__main__":
    
    download_data()
    train_loader, valid_loader, test_loader = load_data()
    
    for images, labels in train_loader:
        plt.imshow(images[0].permute(1, 2, 0))
        plt.title(f"Label: {labels[0]}")
        plt.show()
        break
    
