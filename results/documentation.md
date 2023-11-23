# Validation and training loss is not decreasing

- Data normalization doesn't have weird color anymore like green/red/bue (i just changed the mean and std deviation ) but it doesn't work on dark background with dark skin tone.

- Resized to 256x256 instead of 512x512 (still doesn't work)

- the mnist dataset work on our model (when we mimic the lecture model)

- our dataset doesn't work on the mimiced lecture model

- tried removing all the convo layers basically to just have a simple mlp but it still didn't work

- Avyan made it work! the solution was:

    - Resizing it from all images in dataset from 512px to 32px

    - Changing 3 to 64 kernel in the initial convolutional layer

    - Changing 3 to ... dense layers

    - Changing 8 to 1024 neurons per dense layers

    - Changing 3 to 4 convolutional layers in network  

# Current Status:

- Our best model has 98% testing accuracy

- We want to training it again with rotations, transitions and antialias

- Our model can't predict images from a webcam due to the different pixel distribution from the images in the dataset compare to an imgae taken from a webcam

# Model doesn't work on images taken from a webcam:

## Solution: Domain Adaptation with the CORAL method


