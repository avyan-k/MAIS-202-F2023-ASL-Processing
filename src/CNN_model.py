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
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from itertools import product
from torchsummary import summary



DEVICE = ld.load_device()

class CNN_model(nn.Module):

  def __init__(self,numberConvolutionLayers=3,initialKernels=5,numberDense = 0,neuronsDLayer=0,dropout=0.5,dataset = "ASL"):
    
    super(CNN_model, self).__init__() # calls the constructor of the parent class (nn.Module) to properly initialize the model
    self.network = nn.Sequential(
      nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3,padding="same"),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2),
      
      nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3,padding="same"),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3,padding="same"),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3,padding="same"),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Flatten(start_dim=1),
      nn.Dropout(p = 0.5),
      nn.Linear(2048, 1024),
      nn.BatchNorm1d(1024),
      nn.ReLU(),
      nn.Dropout(p = 0.5),
      nn.Linear(1024, 27)


    )
    # Setting up databases constants
    if dataset == "ASL":
      channels,classes,image_size = (3,27,(32,32))
    elif dataset == "MNIST":
      channels,classes,image_size = (1,10,(28,28))
    else:
      raise AttributeError("Unknown dataset", dataset)
    
    #Convolutional Network
    self.convolutional_network = nn.ModuleList() # list that will store the convolutional layers of the model
    kernelsPerLayers = initialKernels

    self.convolutional_network.append(nn.Conv2d(in_channels=channels,out_channels=kernelsPerLayers, kernel_size=3,padding="same")) #first convolutional layer
    self.convolutional_network.append(nn.BatchNorm2d(kernelsPerLayers)) #batch normalize the data
    for i in range(numberConvolutionLayers - 1):
      self.convolutional_network.append(nn.Conv2d(in_channels=kernelsPerLayers,out_channels=kernelsPerLayers*2, kernel_size=3,padding="same")) #add convolution layer
      kernelsPerLayers *= 2 #double kernels everytime
      self.convolutional_network.append(nn.BatchNorm2d(kernelsPerLayers)) #batch normalize the data


    flattened = int((image_size[0] / (2**(numberConvolutionLayers))))**2 * kernelsPerLayers #Flattened Output of Convolutional Layers
    self.dropout_layer = nn.Dropout(p=dropout) # dropout to reduce model and prevent overfitting

    #Dense Network
    self.dense_network = nn.ModuleList() # list that will store the dense layers of the model
    neurons = neuronsDLayer

    self.dense_network.append(nn.Linear(flattened, neurons)) # first dense layer
    self.dense_network.append(nn.BatchNorm1d(neurons)) #batch normalize the data
    for i in range(numberDense-1): # one dense layer has already been added
      self.dense_network.append(nn.Linear(neurons, neurons))
      self.convolutional_network.append(nn.BatchNorm1d(neurons)) #batch normalize the data

    self.dense_network.append(nn.Linear(neurons, classes)) # classification layer - not counted as part of (hidden)dense network


  def forward(self, x):
    
    '''Forward pass function, needs to be defined for every model'''
    for index,convLayer in enumerate(self.convolutional_network):
      x = convLayer(x) # applies a convolution operation to the input
      if index % 2 == 1: #note that every other layer is batch normalize, we only activate and max pool for convolution
          x = F.relu(x) # activation function
          x = F.max_pool2d(x, 2) # 2x2 maxpool each convolutional layer except the first one


    x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
    x = self.dropout_layer(x) # dropout on some of the convolution

		# fully connected (dense) layers defined in self.dense_network
    for index, dense_layer in enumerate(self.dense_network):
      if index == len(self.dense_network) - 1: #dropout on last layer
        x = self.dropout_layer(x)
        x = dense_layer(x)
      else:
        x = dense_layer(x)
        if index % 2 == 1: #again only activate after batch normalizing
          x = F.relu(x) # ReLU activation function

    return x # returns predicted class probabilities for each input
  

def train_model(cnn,train_loader, valid_loader, test_loader, num_epochs = 200,num_iterations_before_validation = 27,weight_decay=0.00001):
  losses = np.empty(num_epochs)
  start = time.time()
  text_file = open(r"results\training\losses.txt", "w")
  text_file.write(f"Attempting {num_epochs} epochs on date of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
  # text_file.write(f"Model Summary: {summary(cnn, (1, 28, 28))}\n")
  text_file.write(f"{'-'*52}\n")
  text_file.close()

  cnn = cnn.to(DEVICE)

  # Initializes the Adam optimizer with the model's parameters
  # optimizer = optim.SGD(cnn.parameters(), lr,weight_decay=wd)
  optimizer = optim.Adam(cnn.parameters(), lr=0.001,weight_decay=0.01)
  loss = nn.CrossEntropyLoss().to(DEVICE)
  accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=27).to(DEVICE)
  # cnn_models[lr] = cnn
  # cnn_models[wd] = cnn
  
  for epoch in range(num_epochs):
    
    # Iterate through the training data
    for iteration, (X_train, y_train) in enumerate(train_loader):

      # print(y_train)
      optimizer.zero_grad()

      X_train = X_train.to(DEVICE)
      y_train = y_train.to(DEVICE)
      
      # forward pass of the CNN model on the input data to get predictions
      y_hat = cnn(X_train)
      


      optimizer.zero_grad()
      
      # comparing the model's predictions with the truth labels
      train_loss = loss(y_hat, y_train)
      # if iteration % 9 == 0:
      #   print(f"Epoch: {epoch} Iteration: {iteration} Loss: {train_loss}")
      #   print(f"Predictions:\n{y_hat}")
      # backpropagating the loss through the model
      train_loss.backward()

      # takes a step in the direction that minimizes the loss
      optimizer.step()

      if iteration % num_iterations_before_validation  == 0 and epoch > (num_epochs/2):
        with torch.no_grad():

          # Keep track of the losses & accuracies
          val_accuracy_sum = 0
          val_loss_sum = 0

          # Make a predictions on the full validation set, batch by batch
          for X_val, y_val in valid_loader:

            # Move the batch to GPU if it's available
            X_val = X_val.to(DEVICE)
            y_val = y_val.to(DEVICE)

            y_hat = cnn(X_val)
            val_accuracy_sum += accuracy(y_hat, y_val)
            val_loss_sum += loss(y_hat, y_val)

          # Divide by the number of iterations (and move back to CPU)
          val_accuracy = (val_accuracy_sum / len(valid_loader)).cpu()
          val_loss = (val_loss_sum / len(valid_loader)).cpu()

          # Store the values in the dictionary

          # Print to console
          print(f"EPOCH = {epoch} --- ITERATION = {iteration}")
          print(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}")
      #logging results
    training_loss = train_loss.cpu()
    losses[epoch] = training_loss
    print(f"\n\nloss: {training_loss.item()} epoch: {epoch}")
    print("It has now been "+ time.strftime("%Mm%Ss", time.gmtime(time.time() - start))  +"  since the beginning of the program")
    text_file = open(r"results\training\losses.txt", "a")  
    text_file.write(f"loss: {training_loss.item()} epoch: {epoch}\n")
    current =  time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))
    text_file.write(f"It has now been {current} since the beginning of the program\n")
    text_file.close()

    

  text_file = open(r"results\training\losses.txt", "a") 
  text_file.write(f"{losses}")
  text_file.close()
  return losses


if __name__ == "__main__":
  number_of_epochs = 500
  train_loader, valid_loader, test_loader = ld.load_data()
  cnn = CNN_model(numberConvolutionLayers=4,initialKernels=64,numberDense=0,neuronsDLayer=1024,dropout=0.5, dataset="ASL").to(DEVICE)
  summary(cnn, (3, 32, 32))
  losses = train_model(cnn, train_loader, valid_loader, test_loader,num_epochs=number_of_epochs)
  xh = np.arange(0,number_of_epochs)
  plt.plot(xh, losses, color = 'b', 
         marker = ',',label = "Loss") 
  plt.xlabel("Epochs Traversed")
  plt.ylabel("Training Loss")
  plt.grid() 
  plt.legend() 
  plt.show()