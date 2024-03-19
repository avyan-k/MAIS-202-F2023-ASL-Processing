import os
import sys
import mediapipe as mp
import tensorflow as tf
from os.path import dirname, abspath, join
# Get the directory of the current file (app.py) and find the parent directory
current_dir = dirname(abspath(__file__))
parent_dir = dirname(current_dir)

# Add the 'src' directory to the path
sys.path.append(join(parent_dir, 'src'))
import CNN_model as cnn
import loading_dataset as ld
sys.path.append(current_dir)
from flask import Flask, render_template, request
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch import tensor, float32, argmax
from datetime import datetime
import torchmetrics
from glob import glob
from PIL import Image
import predict

DEVICE = ld.load_device()

def croped_shaped_image(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return None
    png_files = glob(os.path.join(folder_path, "*.png"))\

    if not png_files:
        print(f"No PNG files found in folder '{folder_path}'.")
        return None
    png_files.sort(key=os.path.getmtime, reverse=True)
    img = Image.open(png_files[0])
    
    width, height = img.size
    crop_size = min(width, height)
    crop_left = (width - crop_size) // 2 
    crop_top = (height - crop_size) // 2
    crop_right = crop_left + crop_size
    crop_bottom = crop_top + crop_size
    # Crop the image
    cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    # Resize the cropped image to 512x512
    resized_img = cropped_img.resize((512, 512))  # Use antialiasing for better qualityImage.ANTIALIAS
    return resized_img

def getLetter():
    X = croped_shaped_image("images/saved_arrays")
    image_array = np.array(X.convert('RGB'))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
    X_array = ld.image_to_landmarks(mp_image)
    X_array =X_array.astype(np.float32)
    mlp = predict.load_mlp_model()
    prediction = mlp(tensor(X_array.reshape(1,21,2),dtype=float32).to(DEVICE)) # reshape to incorporate batch size
    letter_prediction = argmax(prediction) #letter is of highest probability
    # for x in loader:
    #     print(x)
    #     x = x.to(DEVICE)
    #     letter_prediction = mlp(x) # error here
    return predict.ALPHABET[letter_prediction]

def save_image(image_array, folder_path, filename):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  # Convert the NumPy array to a PIL Image object (assuming RGB format)
  image = Image.fromarray(image_array.astype(np.uint8))

  # Save the image as a PNG file
  image.save(os.path.join(folder_path, filename))

  print(f"Image saved successfully: {os.path.join(folder_path, filename)}")
  

app = Flask(__name__)

@app.route("/") # rendering html page
def index():
    return render_template("index.html", name='Main')

@app.route("/receive",methods=["POST"])
def receive_data():
    data = request.get_json()
    image_data = data.get('imageData', [])
    image_data = np.asarray(image_data)

    image_data = image_data[:, :, :3]# (480, 640, 3)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory_path = 'images/saved_arrays'
    os.makedirs(directory_path, exist_ok=True)
    filename = f"image_data_{timestamp}.png"
    save_image(image_data, directory_path, filename)
    letter = getLetter()
    
    return {'letter': letter}

if __name__ == "__main__":
    app.run(port=5000,debug=True)