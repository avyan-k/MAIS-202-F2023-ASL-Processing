from flask import Flask, render_template, request
import loading_dataset as ld
from CNN_model import MLP_model
import torch

app = Flask(__name__)

@app.route("/") # rendering html page
def index():
    return render_template("index.html", name='Main')

@app.route("/receive",methods=["POST"])
def receive_data():
    
    data = request.get_json()
    pythonConvert = data.get('pythonConvert', [])
    print(pythonConvert)
    return 'Data received successfully'

if __name__ == "__main__":
    app.run(port=8000,debug=True)