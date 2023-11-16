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
    
    self.convolutional_network = nn.ModuleList() # list that will store the convolutional layers of the model
    kernelsPerLayers = initialKernels

    if dataset == "ASL":
      channels,classes,image_size = (3,27,(512,513))
    elif dataset == "MNIST":
      channels,classes,image_size = (1,10,(28,28))
    else:
      raise AssertionError("Unknown dataset", dataset)

    # self.convolutional_network.append(nn.BatchNorm2d(3)) # TODO
    self.convolutional_network.append(nn.Conv2d(in_channels=channels,out_channels=kernelsPerLayers, kernel_size=3,padding="same"))
    
    for index in range(numberConvolutionLayers-1):
      
      self.convolutional_network.append(nn.Conv2d(in_channels=kernelsPerLayers,out_channels=kernelsPerLayers*2, kernel_size=3,padding="same"))
      kernelsPerLayers *= 2 # TODO SWITCH BACK
      
      
		# computes the flattened output size after convolutional and pooling layers
		# 64
    # kernelsPerLayers = 0 # TODO remove, rest for MLP only
    # self.flatten = int(((image_size[0]*image_size[1])/(2**(numberConvolutionLayers))) * kernelsPerLayers)
    self.flatten = int((image_size[0] / (2**(numberConvolutionLayers ))))**2 * kernelsPerLayers

		# store dense (fully connected) layers, both hidden and final classification layers.
    self.dense_network = nn.ModuleList()

    if numberDense > 0:
      #add batch normalization layer
      #self.dense_network.append(nn.BatchNorm1d(self.flatten)) # TODO
      # add the dense layers appropriately
      self.dense_network.append(nn.Linear(self.flatten, neuronsDLayer)) # first dense layer

      for i in range(numberDense - 1): # one dense layer has already been added
        self.dense_network.append(nn.Linear(neuronsDLayer, neuronsDLayer))
        #add batch normalization layer
        # self.dense_network.append(nn.BatchNorm1d(neuronsDLayer)) # TODO
      # classification layer - not counted as part of (hidden)dense network
      self.dense_network.append(nn.Linear(neuronsDLayer, classes))


    else: # only 1 dense layer for classifications - no hidden - default

      self.dense_network.append(nn.Linear(self.flatten, classes))

  def forward(self, x):
    
    '''Forward pass function, needs to be defined for every model'''

    for index,convLayer in enumerate(self.convolutional_network):

      x = convLayer(x) # applies a convolution operation to the input
      x = F.relu(x) # activation function
      x = F.max_pool2d(x, 2) # 2x2 maxpool each convolutional layer except the last one


    x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector

		# fully connected (dense) layers defined in self.dense_network
    for i, dense_layer in enumerate(self.dense_network):
      x = dense_layer(x)
      if i < len(self.dense_network) - 1:
        x = F.relu(x) # ReLU activation function

		# converts the model's raw output into class probabilities
    x = F.softmax(x, dim = 1) # dim = 1 to softmax along the rows of the output

    return x # returns predicted class probabilities for each input
  

def train_model(cnn,train_loader, valid_loader, test_loader, num_epochs = 200,num_iterations_before_validation = 30,weight_decay=0.00001):
  cnn.train() #maybe putting the model in train mode was what we had to do??
  start = time.time()
  text_file = open(r"results\training\losses.txt", "w")
  text_file.write(f"Attempting {num_epochs} epochs on date of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
  # text_file.write(f"Model Summary: {summary(cnn, (1, 28, 28))}\n")
  text_file.write(f"{'-'*52}\n")
  text_file.close()

  cnn = cnn.to(DEVICE)

  # Initializes the Adam optimizer with the model's parameters
  # optimizer = optim.SGD(cnn.parameters(), lr,weight_decay=wd)
  optimizer = optim.Adam(cnn.parameters(), lr=0.1,weight_decay=0.01)
  loss = nn.CrossEntropyLoss().to(DEVICE)
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
      
      # if iteration % 100 == 0:
      #   print(f"Iteration: {iteration} Predictions:\n{y_hat}")
      optimizer.zero_grad()
      
      # comparing the model's predictions with the truth labels
      train_loss = loss(y_hat, y_train)

      # backpropagating the loss through the model
      train_loss.backward()

      # takes a step in the direction that minimizes the loss
      optimizer.step()

      #logging results
    print(f"loss: {train_loss.item()} epoch: {epoch}")
    print("It has now been "+ time.strftime("%Mm%Ss", time.gmtime(time.time() - start))  +"  since the beginning of the program")
    text_file = open(r"results\training\losses.txt", "a")  
    text_file.write(f"loss: {train_loss.item()} epoch: {epoch}\n")
    current =  time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - start))
    text_file.write(f"It has now been {current} since the beginning of the program\n")
      
  return cnn


if __name__ == "__main__":
  train_loader, valid_loader, test_loader = ld.load_data_mnist()
  cnn = CNN_model(numberConvolutionLayers=3,initialKernels=5,numberDense=0,neuronsDLayer=100,dropout=0.5, dataset="MNIST")
  summary(cnn, (1, 28, 28))
  cnn = train_model(cnn, train_loader, valid_loader, test_loader,num_epochs=100)
