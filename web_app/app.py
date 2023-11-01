import io
import torch
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# model = Model()
# model.load_state_dict(torch.load("weights.pth"))

@app.route("/") # rendering html page
def index():
    return render_template("web_app.html")

@app.route("/data")
def data():
    my_dict = {'data' : 1+1 }
    return my_dict
    # im = Image.open(io.BytesIO(request.data)).convert("L")
    # im = transformation(im).unsqueeze(0)
#     with torch.no_grad():
#         preds = model(im)
#         preds = torch.argmax(preds, axis=1)
#         print(preds[0].item())
#         return {"data": preds[0].item()}

if __name__ == "__main__":
    app.run(debug=True) # degubber pin is 238-498-490