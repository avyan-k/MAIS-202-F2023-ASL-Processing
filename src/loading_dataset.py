import os
import torch
from torchvision import datasets, transforms
import opendatasets as od

PATH_TO_DATA = r"synthetic-asl-alphabet"

def download_data():
    if not os.path.exists(r"synthetic-asl-alphabet"):
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data")
        
def load_data():
    train_data_path = PATH_TO_DATA + r"/Test_Alphabet"
    test_data_path = PATH_TO_DATA + r"/Train_Alphabet"
    # Data Normalization
    means = [0.5,0.5,0.5]
    standard_deviations = [0.5,0.5,0.5]
    # Data Augmentation
    randomizing_transforms = [transforms.RandomRotation(30),transforms.RandomHorizontalFlip()]
    # Data Processing
    processing_transforms = [transforms.Resize(255), transforms.ToTensor(), transforms.Normalize(means, standard_deviations)]
    # Loading The Dataset
    train_dataset = datasets.ImageFolder(train_data_path, transform=transforms.Compose(randomizing_transforms + processing_transforms))
    test_dataset = datasets.ImageFolder(test_data_path, transform=transforms.Compose(randomizing_transforms + processing_transforms))
    # Set Batch Size and shuffles the data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    return train_dataset,test_dataset, train_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataset, test_dataset, train_dataloader, test_dataloader = load_data()