import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import torchmetrics
import loading_dataset as ld
import CNN_model


def weight_regularization():
    
    channel_mean = np.mean(train_data, axis=0)
    channel_std = np.std(train_data, axis=0)
    
    norm_train_data = (train_data - channel_mean) / channel_std
    norm_validation_data = (validation_data - channel_mean) / channel_std
    norm_test_data = (test_data - channel_mean) / channel_std
    
    return channel_mean,channel_std,norm_train_data,norm_validation_data,norm_test_data

if __name__ == "__main__":
    weight_regularization()