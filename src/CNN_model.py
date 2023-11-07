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
from sklearn.model_selection import GridSearchCV

train_loader, valid_loader, test_loader = ld.load_data()
DEVICE = ld.load_device()

class CNN_model(nn.Module):

  def __init__(self,numberConv=3,initialKernels=5,numberDense = 0,neuronsDLayer=0,dropout=0.5):
    
    super(CNN_model, self).__init__() # calls the constructor of the parent class (nn.Module) to properly initialize the model
    
    self.convolutional_network = nn.ModuleList() # list that will store the convolutional layers of the model
    kernelsPerLayers = initialKernels
    #add batch normalization layer
    self.convolutional_network.append(nn.BatchNorm2d(3))
    self.convolutional_network.append(nn.Conv2d(in_channels=3,out_channels=kernelsPerLayers, kernel_size=5,padding="same"))
    
    for index in range(numberConv-1):
      
      self.convolutional_network.append(nn.Conv2d(in_channels=kernelsPerLayers,out_channels=kernelsPerLayers*2, kernel_size=5,padding="same"))
      kernelsPerLayers *= 2
      
		# computes the flattened output size after convolutional and pooling layers
		# 64
    self.flatten = int((512 / (2**(min(3,numberConv )))))**2 * initialKernels * 2 **(numberConv-1)
		
		# store dense (fully connected) layers, both hidden and final classification layers.
    self.dense_network = nn.ModuleList()

    if numberDense > 0:
      #add batch normalization layer
      self.dense_network.append(nn.BatchNorm1d(self.flatten))
      # add the dense layers appropriately
      self.dense_network.append(nn.Linear(self.flatten, neuronsDLayer)) # first dense layer

      for i in range(numberDense - 1): # one dense layer has already been added
        self.dense_network.append(nn.Linear(neuronsDLayer, neuronsDLayer))
        #add batch normalization layer
        self.dense_network.append(nn.BatchNorm1d(neuronsDLayer))
      # classification layer - not counted as part of (hidden)dense network
      self.dense_network.append(nn.Linear(neuronsDLayer, 27))


    else: # only 1 dense layer for classifications - no hidden - default

      self.dense_network.append(nn.Linear(self.flatten, 27))

  def forward(self, x):
    
    '''Forward pass function, needs to be defined for every model'''

    for index,convLayer in enumerate(self.convolutional_network):

      x = convLayer(x) # applies a convolution operation to the input
      x = F.relu(x) # activation function

			# 2x2 maxpool each convolutional layer except the last one
      if index <3:
        x = F.max_pool2d(x, 2)

    x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector

		# fully connected (dense) layers defined in self.dense_network
    for dense_layer in self.dense_network:
      x = F.leaky_relu(dense_layer(x)) # Leaky ReLU activation function

		# converts the model's raw output into class probabilities
    x = F.softmax(x, dim = 1) # dim = 1 to softmax along the rows of the output

    return x # returns predicted class probabilities for each input
  

  def train_model(self,train_loader, valid_loader, test_loader, num_epochs = 200,num_iterations_before_validation = 30,weight_decay=0.00001):
    
    start = time.time()

    # hyperparameters:
    lr_values = {0.1, 0.01, 0.001}

		# to store metrics and models for different learning rates during training
    cnn_metrics = {}
    cnn_models = {}

    for lr in lr_values:
			
			# to store training and validation accuracies and losses
      cnn_metrics[lr] = {
          "accuracies": [],
          "losses": []
      }

			# to measure the error between predicted and true class labels
			# loss is moved to GPU
      loss = nn.CrossEntropyLoss().to(DEVICE)

			# to set up for a multiclass classification task with 27 classes
      accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=27).to(DEVICE) # Regular accuracy

      cnn = self.to(DEVICE)

			# Initializes the Adam optimizer with the model's parameters
      optimizer = optim.Adam(cnn.parameters(), lr,weight_decay=weight_decay)
      cnn_models[lr] = cnn
      
      for epoch in range(num_epochs):
        
        # Iterate through the training data
        for iteration, (X_train, y_train) in enumerate(train_loader):

          

          X_train = X_train.to(DEVICE)
          y_train = y_train.to(DEVICE)
					
					# forward pass of the CNN model on the input data to get predictions
          y_hat = cnn(X_train)

					# comparing the model's predictions with the truth labels
          train_loss = loss(y_hat,y_train)
	        # backpropagating the loss through the model
          train_loss.backward()
					# takes a step in the direction that minimizes the loss
          optimizer.step()
          
          # gradients of model's parameters are zeroed to avoid accumulating gradients from previous iterations
          optimizer.zero_grad()
          # validation check
          # if iteration % num_iterations_before_validation == 0:
            
          #   # Disable gradient calculations
          #   with torch.no_grad():

          #     val_accuracy_sum = 0
          #     val_loss_sum = 0

          #     for X_val, y_val in valid_loader:

          #       X_val = X_val.to(DEVICE)
          #       y_val = y_val.to(DEVICE)

          #       y_hat = cnn(X_val)
                
          #       val_accuracy_sum += accuracy(y_hat, y_val)
          #       val_loss_sum += loss(y_hat, y_val)
                
          #     # average validation accuracy and loss
          #     val_accuracy = (val_accuracy_sum / len(valid_loader)).cpu()
          #     val_loss = (val_loss_sum / len(valid_loader)).cpu()

          #     cnn_metrics[lr]["accuracies"].append(val_accuracy)
          #     cnn_metrics[lr]["losses"].append(val_loss)

          #     # log validation metric
          #     text_file = open("results\learning_rate_training.txt", "a") 
          #     text_file.write(f"LR = {lr} --- EPOCH = {epoch} --- ITERATION = {iteration}\n")
          #     text_file.write(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}\n")
          #     text_file.write("It has now been "+ str(time.time() - start) +" seconds since the beginning of the program\n")
          #     print("It has now been "+ time.strftime("%Mm%Ss", time.gmtime(time.time() - start))  +"  since the beginning of the program")
          #     text_file.close() 
        
        # log training metric
        text_file = open("results/training/subset_training.txt", "a")  
        print(f"loss: {train_loss.item()} epoch: {epoch}")
        text_file.write(f"loss: {train_loss.item()} epoch: {epoch}\n")
        current =  time.strftime("%Mm%Ss", time.gmtime(time.time() - start))
        print(f"It has now been {current} since the beginning of the program")
        text_file.write(f"It has now been {current} since the beginning of the program\n")
        
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
  
  cnn = CNN_model(numberDense=10, neuronsDLayer=20)
  cnn_metrics = cnn.train_model(train_loader, valid_loader, test_loader)
  
  # plot_parameter_testing(cnn_metrics, 1000)
  # MODEL_PATH = r"cnn_model"
  # torch.save(cnn.state_dict(), MODEL_PATH)
  # param_grid = {
  #   'weight_decay': [0.0001, 0.001, 0.01], 
  # }
  # model = CNN_model() 
  # grid_search = GridSearchCV(model, param_grid, cv=3)
  # grid_search.fit(X_train, y_train)
  # best_weight_decay = grid_search.best_params_['weight_decay']
  # print(best_weight_decay)