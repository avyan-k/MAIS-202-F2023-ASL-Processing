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

train_loader, val_loader, test_loader = ld.load_data()
DEVICE = ld.load_device()

for images, labels in train_loader:
        plt.imshow(images[0].permute(1, 2, 0))
        plt.title(f"Label: {labels[0]}")
        plt.show()
        break
    

class MLP(nn.Module): #extend nn.Module
  '''
    Class representing a multilayer perceptron with 2 hidden layers of sizes 100, 50 and 1 output layer of size 10 (10 classes)
  '''
  def __init__(self):
    super(MLP, self).__init__()

    self.layer1 = nn.Linear(787968, 100) # Images are 28 * 28 pixels, the first hidden layer has size 100

    ### TODO 1: Create 2 more nn.Linear(input_size, output_size) fields for the class
    ### Call them layer2, and layer3
    ### Hint 1: Layer 2 takes the output of layer 1 as input and outputs size 50
    self.layer2 = nn.Linear(100, 50)
    ### Hint 2: Layer 3 takes the output of layer 2 as input and outputs size 10
    self.layer3 = nn.Linear(50, 27)


  def forward(self, x): #tells super constructor what order the layers need to be in
    '''Forward pass function, needs to be defined for every model'''

    x = torch.flatten(x, start_dim = 1) # Since the inputs are images, we need to flatten them into 1D vectors first

    x = self.layer1(x)
    x = F.relu(x)

    ### TODO 2: Pass x through linear layer 2
    x = self.layer2(x)
    x = F.relu(x)
    ### Then apply Relu to the output
    ### Hint: Look at what we did to pass through layer1 and activate above

    x = self.layer3(x)
    x = F.softmax(x, dim = 1) # dim = 1 to softmax along the rows of the output (We want the probabilities of all classes to sum up to 1)

    return x
  
lr_values = {0.01, 0.1}
num_epochs = 50 # We will go through the whole training dataset 2 times during training
num_iterations_before_validation = 100 # We will compute the validation accuracy every 1000 iterations

loss = nn.CrossEntropyLoss() # Since we are doing multiclass classification
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10) # Regular accuracy

# If a GPU is available, use it
if torch.cuda.is_available():
  DEVICE = torch.device("cuda")

# Else, revert to the default (CPU)
else:
  DEVICE = torch.device("cpu")

print(DEVICE)

loss = nn.CrossEntropyLoss().to(DEVICE) # Since we are doing multiclass classification
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=27).to(DEVICE) # Regular accuracy

# To track the validation metrics for each of the two hyperparameters, as well as the trained model for each
mlp_metrics = {}
mlp_models = {}

for lr in lr_values:

  # Create empty lists of tracking the validation loss / accuracy throughout iterations for this hyperparameter
  mlp_metrics[lr] = {
      "accuracies": [],
      "losses": []
  }

  # Initialize an MLP and attach a Stochastic Gradient Descent optimizer to its weights
  # Note: The optimizer defines the formula to use to update the weights
  # SGD is just plain gradient descent, an improvement is the Adam gradient descent
  mlp = MLP().to(DEVICE) #on cpu by default, but this moves to gpu so that code works for those with gpus
  optimizer = optim.SGD(mlp.parameters(), lr)

  # Store the model inside of the models dictionary
  mlp_models[lr] = mlp

  # Iterate through the epochs - standard training loop
  for epoch in range(num_epochs):

    # Iterate through the training data
    for iteration, (X_train, y_train) in enumerate(train_loader):

      # Move the batch to GPU if it's available
      X_train = X_train.to(DEVICE)
      y_train = y_train.to(DEVICE)

      # The optimizer accumulates the gradient of each weight as we do forward passes -> zero_grad resets all gradients to 0
      optimizer.zero_grad()

      # Compute a forward pass and make a prediction
      # Note that calling mlp(X) is equivalent to calling mlp.foward(X) (Benefit of extending nn.Module)
      y_hat = mlp(X_train)

      # Compute the loss
      train_loss = loss(y_hat, y_train)

      # Compute the gradients in the optimizer
      # Calling "backward" on the loss populates the gradients that the optimizer keeps track of
      train_loss.backward()

      # The step function uses the computed gradients to update the weights
      # It follows the strategy of the specified optimizer (In this case, plain SGD)
      optimizer.step()

      # Check if should compute the validation metrics for plotting later
      if iteration % num_iterations_before_validation == 0:

        # Stop computing gradients on the validation set, we don't want the model to learn from this / just track the loss & accuracy
        with torch.no_grad():

          # Keep track of the losses & accuracies
          val_accuracy_sum = 0
          val_loss_sum = 0

          # Make a predictions on the full validation set, batch by batch
          for X_val, y_val in val_loader:

            # Move the batch to GPU if it's available
            X_val = X_val.to(DEVICE)
            y_val = y_val.to(DEVICE)

            y_hat = mlp(X_val)
            val_accuracy_sum += accuracy(y_hat, y_val)
            val_loss_sum += loss(y_hat, y_val)

          # Divide by the number of iterations (and move back to CPU)
          val_accuracy = (val_accuracy_sum / len(val_loader)).cpu()
          val_loss = (val_loss_sum / len(val_loader)).cpu()

          # Store the values in the dictionary
          mlp_metrics[lr]["accuracies"].append(val_accuracy)
          mlp_metrics[lr]["losses"].append(val_loss)

          # Print to console
          print(f"LR = {lr} --- EPOCH = {epoch} --- ITERATION = {iteration}")
          print(f"Validation loss = {val_loss} --- Validation accuracy = {val_accuracy}")