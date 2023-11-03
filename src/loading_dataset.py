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
TRAIN_SET_SIZE = 24300
TESTING_SET_SIZE = 2700

def download_data():
    
    if not os.path.exists(r"synthetic-asl-alphabet"):
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data")
        
def load_data():

    # Data Normalization
    means = [0.4916,0.4697,0.4251]
    standard_deviations = [0.1584,0.1648,0.1768]
    
    # Data Augmentation
    randomizing_transforms = [transforms.RandomRotation(15),transforms.RandomHorizontalFlip()]
   
    # Transform the data to torch tensors and normalize it
    processing_transforms = [transforms.ToTensor()]
    #processing_transforms = [transforms.ToTensor(), transforms.Normalize(means, standard_deviations)]
    
    # Loading The Dataset
    full_train_dataset = datasets.ImageFolder(train_data_path, transform=transforms.Compose(randomizing_transforms + processing_transforms))
    test_dataset = datasets.ImageFolder(test_data_path, transform=transforms.Compose(processing_transforms))
    
    # Split the datasets into training, validation, and test sets
    train_dataset, valid_dataset, _ = random_split(full_train_dataset, [int(TRAIN_SET_SIZE*0.75), int(TRAIN_SET_SIZE*0.25),0])
    test_dataset, _, _ = random_split(test_dataset, [TESTING_SET_SIZE, len(test_dataset) - TESTING_SET_SIZE, 0])
    
    # Set Batch Size and shuffles the data
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=50, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader

def load_device():
    
    # if a macOs then use mps
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    # elif a GPU is available, use it
    elif torch.cuda.is_available(): 
        device = torch.device("cuda")
    # Else, revert to the default (CPU)
    else: 
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
    
