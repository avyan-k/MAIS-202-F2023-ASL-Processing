import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset,DataLoader, random_split
import opendatasets as od
import matplotlib.pyplot as plt

PATH_TO_DATA = r"synthetic-asl-alphabet"
test_data_path = PATH_TO_DATA + r"/Test_Alphabet"
train_data_path = PATH_TO_DATA + r"/Train_Alphabet"
TRAIN_SET_SIZE = 24300
TESTING_SET_SIZE = 2700

def download_data():
    
    if not os.path.exists(r"synthetic-asl-alphabet"):
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data")


def load_data():
    
    # Define the transformations, including normalization
    processing_transforms = v2.Compose([
            # v2.RandomAffine(degrees = 15,translate = (0.15,0.15)),
            # v2.Grayscale(num_output_channels = 3),
            v2.Resize(32),
            v2.ToTensor(), # Convert PIL Image to tensor
            
    ])
    
    # Apply Transformation and Loading Dataset
    full_train_dataset = datasets.ImageFolder(train_data_path,transform=processing_transforms)
    test_dataset = datasets.ImageFolder(test_data_path,transform=processing_transforms)
    
    # Split the datasets into training, validation, and testing sets
    train_dataset, valid_dataset, _ = random_split(full_train_dataset, [int(TRAIN_SET_SIZE*0.9), int(TRAIN_SET_SIZE*0.1),0])
    test_dataset, _, _ = random_split(test_dataset, [TESTING_SET_SIZE, len(test_dataset) - TESTING_SET_SIZE, 0])
    

    # Set Batch Size and shuffles the data
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader

def load_device():
    
    # if a macOs then use mps
    if torch.backends.mps.is_built(): device = torch.device("mps")
    
    # elif a GPU is available, use it
    elif torch.cuda.is_available(): device = torch.device("cuda")
    
    # Else, revert to the default (CPU)
    else: device = torch.device("cpu")
        
    return device

def show_images(loader):
    counter=0
    for images, labels in loader:
        if counter == 5 : break
        plt.imshow(images[counter].permute(1, 2, 0))
        plt.title(f"Label: {labels[0]}")
        plt.show()
        counter+=1

if __name__ == "__main__":
    
    download_data()
    train_loader, valid_loader, test_loader = load_data()
    show_images(train_loader)