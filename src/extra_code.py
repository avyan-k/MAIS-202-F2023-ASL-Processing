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