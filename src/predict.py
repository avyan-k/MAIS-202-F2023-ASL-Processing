import os
from PIL import Image
from MLP_model import MLP_model
import loading_dataset as ld
import torch
import numpy as np
import matplotlib.pyplot as plt

ALPHABET = ['A', 'B', ' ', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
DEVICE = ld.load_device()

def load_mlp_model():
    model = MLP_model(layers = 5, neurons_per_layer = 64,dropout=0, input_shape = (21,2)).to(DEVICE)
    model.load_state_dict(torch.load(r"our_models/MLP/model3.pt",map_location = DEVICE))
    model.eval()
    return model

def convert():
    image_files = []
    folder_path = 'images/saved_arrays'
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        modification_time = os.path.getmtime(filepath)
        image_files.append((filepath, modification_time))

    # Sort the list of image files by modification time in descending order
    image_files.sort(key=lambda x: x[1], reverse=True)

    most_recent_image_path, _ = image_files[0]
    print("Most recently saved image:", most_recent_image_path)
        
    image_data = np.load(most_recent_image_path)
    plt.imshow(image_data)
    plt.axis('off')  # Turn off axis numbering
    plt.savefig('image.png')
    
    image = Image.fromarray(image_data, 'RGB')
    image.save('image.png')
    image.show()
    

# def predicted():
    
if __name__ == "__main__":
    convert()