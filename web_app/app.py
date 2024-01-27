import os
from flask import Flask, render_template, request
import numpy as np
from datetime import datetime


app = Flask(__name__)

@app.route("/") # rendering html page
def index():
    return render_template("index.html", name='Main')

@app.route("/receive",methods=["POST"])
def receive_data():
    data = request.get_json()
    image_data = data.get('imageData', [])
    print(len(image_data))
    image_data = np.asarray(image_data)
    
    # For accessing the color array outside, you could save it to a file or pass it to another function
    image_data = image_data.reshape((512, 512, 4))[:, :, :3]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory_path = 'images/saved_arrays'
    os.makedirs(directory_path, exist_ok=True)
    filename = f"{directory_path}/image_data_{timestamp}.npy"
    
    # Save the array to the file
    np.save(filename, image_data)
    letter = "A"
    return {'letter': letter}

if __name__ == "__main__":
    app.run(port=5000,debug=True)