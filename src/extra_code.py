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
# ============= FROM MLP_model.py ==============================

  # test_dict = {}
  # for filename in os.listdir("our_models"):
  #   model_path = os.path.join("our_models", filename)
  #   if os.path.isfile(model_path) and model_path.endswith('.pt'):
  #     cnn = CNN_model(numberConvolutionLayers=4,initialKernels=64,numberDense=0,neuronsDLayer=1024,dropout=0.5).to(DEVICE)
  #     cnn.load_state_dict(torch.load(model_path, map_location = DEVICE))
  #     print(test(cnn, test_loader))
  #     test_dict[model_path] = test(cnn, test_loader)
  # print(max(test_dict, key=test_dict.get))
  # cnn = CNN_model(numberConvolutionLayers=4,initialKernels=64,numberDense=0,neuronsDLayer=1024,dropout=0.5,classes=29,image_size=(128,128)).to(DEVICE)
    
      # best_losses = train_model(model=mlp,input_shape=(1, 2, 21, 1),train_loader=landmark_train, valid_loader=landmark_validation,num_epochs=number_of_epochs,number_of_validations = 3,learning_rate=0.001)

  # best_loss = train_model(model=mlp,input_shape=(1, 3, 22, 1),train_loader=landmark_train, valid_loader=landmark_validation,num_epochs=number_of_epochs,num_iterations_before_validation = 2430,learning_rate=0.05)
  # best_losses = train_model(model=cnn,input_shape=(1, 3, 128,128),train_loader=train_loader, valid_loader=valid_loader,num_epochs=number_of_epochs,num_iterations_before_validation = 2430)
  # best_lr, best_nbr_layers, best_neurons_per_layers = 0, 0, 0
  # best_loss = float('-inf')
  # best_losses = []
  # grids = 10
  # for i in tqdm(range (grids),desc="Learning Rate",position=0):
  #       for j in tqdm(range(grids),desc="Layers",position=1,leave=False):
  #             for k in tqdm(range(grids),desc="Neurons",position=2,leave=False):
  #                     lr = np.divide(i+1,100)#such that learning rate is at most 0.1, close to 0
  #                     layers = j
  #                     neurons = (k+1)*10 #neurons from 1 to grids, strictly positive
  #                     mlp = MLP_model(layers = layers, neurons_per_layer = neurons,dropout=0.5, input_shape = (22,3)).to(DEVICE)
  #                     # print(f"learning_rate: {lr} layers: {layers} neurons: {neurons}")
  #                     losses = train_model(model=mlp,input_shape=(1, 3, 22, 1),train_loader=landmark_train, valid_loader=landmark_validation,num_epochs=number_of_epochs,num_iterations_before_validation = 25,learning_rate=lr)
  #                     loss = (stats.trim_mean(losses[10:],0.2))-(stats.trim_mean(losses[:10],0.2))
  #                     if loss > best_loss:
  #                               best_loss = loss
  #                               best_losses = losses
  #                               best_lr = lr
  #                               best_nbr_layers = layers
  #                               best_neurons_per_layers = neurons
  # print(f"Best Parameters: \n learning_rate: {best_lr} layers: {best_nbr_layers} neurons: {best_neurons_per_layers}")
  # to plot the losses
  # xh = np.arange(0,number_of_epochs)
  # plt.plot(xh, best_losses, color = 'b', 
  #        marker = ',',label = "Loss") 
  # plt.xlabel("Epochs Traversed")
  # plt.ylabel("Training Loss")
  # plt.grid() 
  # plt.legend() 
  # plt.show()
  
  # ========= THE OLD CNN MODEL ==========
#   class CNN_model(nn.Module):

#   def __init__(self,numberConvolutionLayers=4,initialKernels=64,numberDense = 0,neuronsDLayer=0,dropout=0.5,channels = 3, classes = 27,image_size = (32,32)):
    
#     # Calls the constructor of the parent class (nn.Module) to properly initialize the model
#     super(CNN_model, self).__init__() 

#     # Convolutional Network
#     self.convolutional_network = nn.ModuleList() # list that will store the convolutional layers of the model
    
    
#     if numberConvolutionLayers > 0: #only add conv layer if needed
#       kernelsPerLayers = initialKernels
#         # First convolutional layer
#       self.convolutional_network.append(nn.Conv2d(in_channels=channels,out_channels=kernelsPerLayers, kernel_size=3,padding="same")) 
#       self.convolutional_network.append(nn.BatchNorm2d(kernelsPerLayers)) # batch normalize the data
      
#       for i in range(numberConvolutionLayers - 1):
#         # Add convolution layer
#         self.convolutional_network.append(nn.Conv2d(in_channels=kernelsPerLayers,out_channels=kernelsPerLayers*2, kernel_size=3,padding="same")) 
#         kernelsPerLayers *= 2 # double kernels everytime
#         self.convolutional_network.append(nn.BatchNorm2d(kernelsPerLayers)) # batch normalize the data
#       # Flattened Output of Convolutional Layers
#       flattened = int((image_size[0] / (2**(numberConvolutionLayers))))**2 * kernelsPerLayers 
#     else:
#       flattened = image_size[0]*image_size[1] * channels # if no conv layers, then flattened is the neurons amount
    
     
#     self.dropout_layer = nn.Dropout(p=dropout) # dropout to reduce model and prevent overfitting

#     # Dense Network
#     self.dense_network = nn.ModuleList() # list that will store the dense layers of the model
#     neurons = neuronsDLayer

#     self.dense_network.append(nn.Linear(flattened, neurons)) # first dense layer
#     self.dense_network.append(nn.BatchNorm1d(neurons)) # batch normalize the data
    
#     for i in range(numberDense-1): # one dense layer has already been added
#       self.dense_network.append(nn.Linear(neurons, neurons)) # add dense layer
#       self.dense_network.append(nn.BatchNorm1d(neurons)) # batch normalize the data

#     self.dense_network.append(nn.Linear(neurons, classes)) # classification layer - not counted as part of (hidden)dense network

#   def forward(self, x):
    
#     '''Forward pass function, needs to be defined for every model'''
#     for index,convLayer in enumerate(self.convolutional_network):
#       # applies a convolution operation to the input
#       x = convLayer(x) 
#       # max pool and relu every other conv
#       if index % 2 == 1: 
#         x = F.relu(x)  
#         x = F.max_pool2d(x, 2) 
#     x = torch.flatten(x, start_dim = 1) # Flatten to a 1D vector
#     x = self.dropout_layer(x) # dropout on some of the convolution

# 		# fully connected (dense) layers defined in self.dense_network
#     for index, dense_layer in enumerate(self.dense_network):
#       # dropout on last layer: no ReLU
#       if index == len(self.dense_network) - 1: 
#         x = self.dropout_layer(x)
#         x = dense_layer(x)
#       else:
#         x = dense_layer(x)
#         # relu every other layer
#         if index % 2 == 1:
#           x = F.relu(x) 

#     return x # returns predicted class probabilities for each input
  
  # =========== capture_input.py =============
import cv2
import os
import time
from PIL import Image
from torchvision import datasets, transforms
from MLP_model import CNN_model
from MLP_model import MLP_model
import loading_dataset as ld
import torch
import torch.nn.functional as F
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from torchinfo import summary

ALPHABET = ['A', 'B', ' ', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
DEVICE = ld.load_device()

def load_mlp_model():
    model = MLP_model(layers = 5, neurons_per_layer = 64,dropout=0, input_shape = (21,2)).to(DEVICE)
    model.load_state_dict(torch.load(r"our_models/MLP/model3.pt",map_location = DEVICE))
    model.eval()
    return model

def capture_images():
    
    # Create a directory with a timestamp to save the captured images
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join("captured_images", timestamp)
    os.makedirs(save_dir)
    
    # Create a directory to save the captured images
    if not os.path.exists("captured_images"):
        os.makedirs("captured_images")

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Set the resolution to 512x512
    cap.set(3, 512)  # Width
    cap.set(4, 512)  # Height
    
    # Allow the camera to adjust for the first image
    time.sleep(2)
    
    mlp = load_mlp_model()
    i=0
    # Capture frame_num of images
    while(True):
        
        choice = input(f"Preparing to capture image {i+1}. Press Enter to capture or Press Space+Enter to stop:")
        
        if choice==" ":
            print("Ending ... ")
            break

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture an image.")
            i+=1
            continue

        predict_image(save_dir,frame,i,mlp)
        
        # Wait for 1 second
        cv2.waitKey(1000)
        i+=1

    # Release the webcam and close OpenCV
    cap.release()
    cv2.destroyAllWindows()

def predict_image(save_dir,frame,i,cnn):
    
    image_filename = os.path.join(save_dir, f"image_{i}.png")
    cv2.imwrite(image_filename, frame)
    
    path = os.path.join(save_dir, f"image_{i}.png")
    img = Image.open(path)
    image = mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(img))
    result = ld.image_to_landmarks(image)
    result = torch.tensor(result.reshape(1,21,2),dtype=torch.float32).to(DEVICE)
    cnn.eval()
    
    output = cnn(result)
    
    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    
    # to check how the output looks like: its giving a bunch of negative numbers ...
    # print(output)
    
    print(f"Image {i + 1} captured and saved as {image_filename}")
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Predicted letter: {ALPHABET[predicted_class.item()]}")
    
    
if __name__ == "__main__":
    capture_images()