import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import opendatasets as od
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from hyperpara_tuning import find_mean_stds

PATH_TO_DATA = r"synthetic-asl-alphabet"
test_data_path = PATH_TO_DATA + r"/Test_Alphabet"
train_data_path = PATH_TO_DATA + r"/Train_Alphabet"
TRAIN_SET_SIZE = 24300
TESTING_SET_SIZE = 2700

def download_data():
    
    if not os.path.exists(r"synthetic-asl-alphabet"):
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data")

def load_data_mnist():
    # Transform the data to torch tensors and normalize it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and standard deviation of the dataset to use for normalization
    ])

    # Preparing the training, validation, and test sets
    train_and_val_sets = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transform) # Will be split into training and validation for hyperparameter tuning
    train_set, val_set,_ = torch.utils.data.random_split(train_and_val_sets, [5000, 1000,54000]) # 55000 training examples, 5000 validation examples
    test_set = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)

    # Prepare loaders which will give us the data batch by batch (Here, batch size is 8)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True)

    # Print the size of each of the datasets
    return train_loader, val_loader, test_loader

def load_data():

    # Data Normalization
    
    # means = [0.4916,0.4697,0.4251] # tracy's bad calculations, can probably discare
    # standard_deviations = [0.1584,0.1648,0.1768]
    
    # means = [0.5,0.5,0.5]
    # standard_deviations = [0.5,0.5,0.5]
    
    # means = [0.4916, 0.4699, 0.4255] # after hyper para tuning gives very intense pics
    # standard_deviations = [0.0251, 0.0265, 0.0278]
    
    train_means = [0.4904, 0.4686, 0.4246] # gives descent pics but pick black for dark background with dark skin tones
    train_standard_deviations = [0.2473, 0.2526, 0.2713]

    test_means = [0.4904, 0.4686, 0.4246]

    test_standard_deviations = [0.2473, 0.2526, 0.2713]
    
    # Define the transformations, including normalization
    train_processing_transforms = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor
        # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((32,32)),
        
    ])
    test_processing_transforms = transforms.Compose([
        transforms.ToTensor(),                             # Convert PIL Image to tensor
        transforms.Normalize(test_means, test_standard_deviations),   # Normalize the image
    ])
    
    full_train_dataset = datasets.ImageFolder(train_data_path,transform=train_processing_transforms)
    test_dataset = datasets.ImageFolder(test_data_path,transform=test_processing_transforms)

    
    # # Loading The Dataset
    # full_train_dataset = datasets.ImageFolder(train_data_path, transform=transforms.Compose(processing_transforms))
    # test_dataset = datasets.ImageFolder(test_data_path, transform=transforms.Compose(processing_transforms))
    
    # Split the datasets into training, validation, and test sets
    train_dataset, valid_dataset, _ = random_split(full_train_dataset, [int(TRAIN_SET_SIZE*0.01), int(TRAIN_SET_SIZE*0.01),int(TRAIN_SET_SIZE*0.98)])
    test_dataset, _, _ = random_split(test_dataset, [TESTING_SET_SIZE, len(test_dataset) - TESTING_SET_SIZE, 0])
    

    # Set Batch Size and shuffles the data
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True)
    
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
    counter=0
    
    for images, labels in train_loader:
        if counter == 5 : break
        # print(images[counter])
        plt.imshow(images[counter].permute(1, 2, 0))
        plt.title(f"Label: {labels[0]}")
        plt.show()
        counter+=1
    
