import io
import torch
from flask import Flask, render_template, request
from PIL import Image
import string
import random

app = Flask(__name__)

@app.route('/')
def home():
    
    letter = random.choice(string.ascii_letters)
    return render_template('index.html', letter=letter)

if __name__ == "__main__":
    app.run(debug=True)
