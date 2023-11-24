import cv2
import os
import time
from PIL import Image
from torchvision import datasets, transforms
from CNN_model import CNN_model
import loading_dataset as ld
import torch
import torch.nn.functional as F

ALPHABET = ['A', 'B', ' ', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
DEVICE = ld.load_device()

def load_cnn_model():
    
    # Load your CNN model here
    model = CNN_model(numberConvolutionLayers=4,initialKernels=64,numberDense=0,neuronsDLayer=1024,dropout=0.5, dataset="ASL").to(DEVICE)
    
    # Load the pretrained weights if available
    model.load_state_dict(torch.load(r"our_models/model1.pt",map_location = DEVICE))
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
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Set the resolution to 512x512
    cap.set(3, 512)  # Width
    cap.set(4, 512)  # Height
    
    # Allow the camera to adjust for the first image
    time.sleep(2)
    cnn = load_cnn_model()
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
        
        predict_image(save_dir,frame,i,cnn)
        
        # Wait for 1 second
        cv2.waitKey(1000)
        i+=1

    # Release the webcam and close OpenCV
    cap.release()
    cv2.destroyAllWindows()

def predict_image(save_dir,frame,i,cnn):
    
    # Save the image with a unique filename
    image_filename = os.path.join(save_dir, f"image_{i}.png")
    cv2.imwrite(image_filename, frame)

    # To test on user input image
    path = os.path.join(save_dir, f"image_{i}.png")
    
    # to test on dataset
    # path = os.path.join(r"images/O.png")

    img = Image.open(path)
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32,antialias=True),  # Resize to 32x32
    ])

    # Apply Transformations
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Make predictions using the CNN model
    cnn.eval() 
     
    # with torch.no_grad():
    output = cnn(img_tensor.to(DEVICE))

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)
    
    # to check how the output looks like: its giving a bunch of negative numbers ...
    # print(output)
    
    print(f"Image {i + 1} captured and saved as {image_filename}")
    print(f"Predicted class: {predicted_class.item()}")
    print(f"Predicted letter: {ALPHABET[predicted_class.item()]}")
    
    
if __name__ == "__main__":
    capture_images()