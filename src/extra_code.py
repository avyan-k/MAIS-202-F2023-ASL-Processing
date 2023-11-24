# ## Tuning Weight Regularization
# With tune_mean_std() in hyperpara_tuning.py we get the average weight for the mean and standard deviation for the each datasets. 

# After approximating the best weight from it we get:

# - mean: [0.4916,0.4697,0.4251]
# - standard deviation: [0.1584,0.1648,0.1768]


# def load_data_mnist():
#     # Transform the data to torch tensors and normalize it
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)) # Mean and standard deviation of the dataset to use for normalization
#     ])

#     # Preparing the training, validation, and test sets
#     train_and_val_sets = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transform) # Will be split into training and validation for hyperparameter tuning
#     train_set, val_set,_ = torch.utils.data.random_split(train_and_val_sets, [5000, 1000,54000]) # 55000 training examples, 5000 validation examples
#     test_set = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)

#     # Prepare loaders which will give us the data batch by batch (Here, batch size is 8)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True)

#     # Print the size of each of the datasets
#     return train_loader, val_loader, test_loader

    
    # # Setting up databases constants
    # if dataset == "ASL":
    #   channels,classes,image_size = (3,27,(32,32))
    # elif dataset == "MNIST":
    #   channels,classes,image_size = (1,10,(28,28))
    # else:
    #   raise AttributeError("Unknown dataset", dataset)
    
    
    # out_channels = initialKernels
    # in_channels = channels
    # self.convolutional_network.append(nn.Conv2d(in_channels=channels,out_channels=out_channels, kernel_size=3,padding="same")) # first convolutional layer
    # self.convolutional_network.append(nn.BatchNorm2d(out_channels)) # batch normalize the data
    
    #  for i in range(numberConvolutionLayers-1):
    #   print(f"{in_channels}")
    #   # Add convolution layer
    #   self.convolutional_network.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding="same"))
    #   self.convolutional_network.append(nn.BatchNorm2d(out_channels)) # batch normalize the data
    #   in_channels = out_channels
    #   out_channels *=2
      
    #   print(f"{out_channels}")