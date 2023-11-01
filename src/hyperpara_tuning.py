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


def weight_regularization(mean,std_devia):
    return mean,std_devia