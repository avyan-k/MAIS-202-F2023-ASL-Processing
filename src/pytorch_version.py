import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import datasets, transforms


PATH_TO_DATA = r"synthetic-asl-alphabet"

def load_data():
    train_data_path = PATH_TO_DATA + r"/Test_Alphabet"
    test_data_path = PATH_TO_DATA + r"/Train_Alphabet"

    means = [0.5,0.5,0.5]
    standard_deviations = [0.5,0.5,0.5]
    randomizing_transforms = [transforms.RandomRotation(30),transforms.RandomHorizontalFlip()]
    processing_transforms = [transforms.Resize(255), transforms.ToTensor(), transforms.Normalize(means, standard_deviations)]

    train_dataset = datasets.ImageFolder(train_data_path, transform=transforms.Compose(randomizing_transforms + processing_transforms))
    test_dataset = datasets.ImageFolder(test_data_path, transform=transforms.Compose(randomizing_transforms + processing_transforms))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    return train_dataset,test_dataset, train_dataloader, test_dataloader


