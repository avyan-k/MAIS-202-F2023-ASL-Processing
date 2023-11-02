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
import time

train_loader, valid_loader, test_loader = ld.load_data()
DEVICE = ld.load_device()

class CNN_model(nn.Module):
  '''
    Class representing a CNN with 2 (convolutional + activation + maxpooling) layers, connected to a single linear layer for prediction
  '''
  def __init__(self,numberConv=3,initialKernels=5,numberDense = 0,neuronsDLayer=0,dropout=0.5):
    
    super(CNN_model, self).__init__()
    
    self.convolutional_network = nn.ModuleList()
    kernelsPerLayers = initialKernels
    self.convolutional_network.append(nn.Conv2d(in_channels=3,out_channels=kernelsPerLayers, kernel_size=5,padding="same"))
    
    for index in range(numberConv-1):
      
      self.convolutional_network.append(nn.Conv2d(in_channels=kernelsPerLayers,out_channels=kernelsPerLayers*2, kernel_size=5,padding="same"))
      kernelsPerLayers *= 2
      
    self.flatten = int((512 / (2**(numberConv if numberConv<3 else 3))))**2 * initialKernels * 2 **(numberConv-1)

    self.linear = nn.Linear(self.flatten, 27) 

  def forward(self, x):
    
    '''Forward pass function, needs to be defined for every model'''
    for index,convLayer in enumerate(self.convolutional_network):
      x = convLayer(x)
      x = F.relu(x)
      if index <3:
        x = F.max_pool2d(x, 2)
     # 2x2 maxpool

    x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
    x = self.linear(x)
    x = F.softmax(x, dim = 1) # dim = 1 to softmax along the rows of the output (We want the probabilities of all classes to sum up to 1)

    return x
  

def train_model(train_loader, valid_loader, test_loader, num_epochs = 2,num_iterations_before_validation = 30):
  
  start = time.time()
  # hyperparameters
  lr_values = {0.01, 0.001}
  cnn_metrics = {}
  cnn_models = {}

  for lr in lr_values:

    cnn_metrics[lr] = {
        "accuracies": [],
        "losses": []
    }

    loss = nn.CrossEntropyLoss().to(DEVICE)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=27).to(DEVICE) # Regular accuracy


    cnn = CNN_model().to(DEVICE)
    optimizer = optim.Adam(cnn.parameters(), lr)
    cnn_models[lr] = cnn
    
    for epoch in range(num_epochs):

      # Iterate through the training data
      for iteration, (X_train, y_train) in enumerate(train_loader):
        # Move the batch to GPU if it's available
        X_train = X_train.to(DEVICE)
        y_train = y_train.to(DEVICE)

        optimizer.zero_grad()

        y_hat = cnn(X_train)

        train_loss = loss(y_hat, y_train)
        
        train_loss.backward()

        optimizer.step()

        if iteration % num_iterations_before_validation == 0:

          with torch.no_grad():

            val_accuracy_sum = 0
            val_loss_sum = 0

            for X_val, y_val in valid_loader:

              X_val = X_val.to(DEVICE)
              y_val = y_val.to(DEVICE)

              y_hat = cnn(X_val)
              val_accuracy_sum += accuracy(y_hat, y_val)
              val_loss_sum += loss(y_hat, y_val)

            val_accuracy = (val_accuracy_sum / len(valid_loader)).cpu()
            val_loss = (val_loss_sum / len(valid_loader)).cpu()

            cnn_metrics[lr]["accuracies"].append(val_accuracy)
            cnn_metrics[lr]["losses"].append(val_loss)

            text_file = open("results\learning_rate_training.txt", "a") 
            text_file.write(f"LR = {lr} --- EPOCH = {epoch} --- ITERATION = {iteration}\n")
            text_file.write(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}\n")
            text_file.write("It has now been "+ str(time.time() - start) +" seconds since the beginning of the program\n")
            print("It has now been "+ str(time.time() - start) +" seconds since the beginning of the program")
            text_file.close()  
  return cnn_metrics
          
          
def plot_parameter_testing(cnn_metrics,num_iterations_before_validation):
  x_axis = np.arange(0, len(cnn_metrics[0.1]["accuracies"]) * num_iterations_before_validation, num_iterations_before_validation)
  # Plot the accuracies as a function of iterations
  plt.plot(x_axis, cnn_metrics[0.01]["accuracies"], label = "Validation accuracies for lr = 0.01")
  plt.plot(x_axis, cnn_metrics[0.001]["accuracies"], label = "Validation accuracies for lr = 0.001")
  plt.xlabel("Iteration")
  plt.ylabel("Validation accuracy")
  plt.title("Validation accuracy as a function of iteration for CNN")
  plt.legend()

if __name__ == "__main__":
  cnn_metrics = train_model(train_loader, valid_loader, test_loader)
  
  cnn_metrics, cnn = train_model(train_loader, valid_loader, test_loader)
  plot_parameter_testing(cnn_metrics, 1000)
  MODEL_PATH = r"cnn_model"
  torch.save(cnn.state_dict(), MODEL_PATH)