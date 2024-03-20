import os
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset,DataLoader, random_split
import opendatasets as od
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pathlib
from tqdm import tqdm
import PIL as pil

PATH_TO_DATA = r"synthetic-asl-alphabet"
PATH_TO_LANDMARK_DATA = r"hand-landmarks"
PATH_TO_SIMPLEASL_DATA = r"asl-alphabet"
test_data_path = PATH_TO_DATA + r"/Test_Alphabet"
train_data_path = PATH_TO_DATA + r"/Train_Alphabet"
TRAIN_SET_SIZE = 24300
TESTING_SET_SIZE = 2700

def download_data():
    
    if not os.path.exists(r"synthetic-asl-alphabet"):
        od.download("https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data")
    if not os.path.exists(r"asl-alphabet"):
        od.download("https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        
def load_data():
    
    # Define the transformations, including normalization
    processing_transforms = v2.Compose([
            # v2.RandomAffine(degrees = 15,translate = (0.15,0.15)),
            # v2.Grayscale(num_output_channels = 3),
            v2.Resize(32),
            v2.ToTensor(), # Convert PIL Image to tensor
            
    ])
    
    # Apply Transformation and Loading Dataset
    full_train_dataset = datasets.ImageFolder(train_data_path,transform=processing_transforms)
    test_dataset = datasets.ImageFolder(test_data_path,transform=processing_transforms)
    
    # Split the datasets into training, validation, and testing sets
    train_dataset, valid_dataset, _ = random_split(full_train_dataset, [int(TRAIN_SET_SIZE*0.01), int(TRAIN_SET_SIZE*0.01),TRAIN_SET_SIZE- int(TRAIN_SET_SIZE*0.01) - int(TRAIN_SET_SIZE*0.01)])
    test_dataset, _, _ = random_split(test_dataset, [TESTING_SET_SIZE, len(test_dataset) - TESTING_SET_SIZE, 0])
    

    # Set Batch Size and shuffles the data
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader

def load_simpleASL_data():
    
    # Define the transformations, including normalization
    processing_transforms = v2.Compose([
            v2.RandomAffine(degrees = 15,translate = (0.15,0.15)),
            v2.Grayscale(num_output_channels = 3),
            v2.Resize(128),
            v2.ToTensor(), # Convert PIL Image to tensor
    ])
    
    # Apply Transformation and Loading Dataset
    full_train_dataset = datasets.ImageFolder(r"asl-alphabet\asl_alphabet_train",transform=processing_transforms)
    test_dataset = datasets.ImageFolder(r"asl-alphabet\asl_alphabet_test",transform=processing_transforms)
    train_size = len(full_train_dataset)
    test_size = len(test_dataset)
    # Split the datasets into training, validation, and testing sets
    train_dataset, valid_dataset, _ = random_split(full_train_dataset, [int(train_size*0.9), int(train_size*0.1),train_size- int(train_size*0.9) - int(train_size*0.1)])
    test_dataset, _, _ = random_split(test_dataset, [test_size, 0, 0])
    

    # Set Batch Size and shuffles the data
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")
    
    return train_loader, valid_loader, test_loader


class HandLandmarksDataset(Dataset):
    def __init__(self, datapath : str,  transform=None):
        """
        Arguments:
            datapath (string): Directory in which the class folders are located.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.paths = list(pathlib.Path(datapath).glob("*/*.npz")) # loads all possible numpy arrays in directory as filepaths
        self.transform = transform
        self.classes =  sorted(entry.name for entry in os.scandir(datapath) if entry.is_dir()) # finds all possible classes and sorts them
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)} # converts a class ot its index, used it __getitem__

    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index : int) -> tuple[torch.Tensor, int]:
        array = np.load(str(self.paths[index]))["arr_0"] # load array from filepath, note that since no arg is provided when saving, the first array is arr_0
        tensor = torch.from_numpy(array) # convert to tensor
        class_name  = self.paths[index].parent.name # since we use pathlib.Path, we can call its parent for the class
        cindex = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(tensor).float(), cindex
        else:
            return tensor.float(), cindex
        
def image_to_landmarks(image):
    
    '''
    Takes an image in MediaPipe format, detects landmarks, and return 21 2d coordinates as an array
    '''
    # creating HandMarker Object from Google MediaPipe
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    detection_result = detector.detect(image) # Detect hand landmarks from the input image.
    image_array = np.zeros((21,2)) #22 landmarks including handedness, 3 coordinates per landmark  
    if  detection_result.hand_landmarks:
        for i,landmark in enumerate(detection_result.hand_landmarks[0]): # iterate through each of the 21 landmarks
            image_array[i][0] = landmark.x # x coordinate of handLandmark
            image_array[i][1] = landmark.y # y coordinate of handLandmark
    return image_array

def save_landmarks_disk():
    landmark_transforms = v2.Compose([
        v2.RandomAffine(degrees = 15,translate = (0.15,0.15)),
    ])
    train_imagepath = os.path.join(PATH_TO_DATA,r"Train_Alphabet") # path to load from
    train_datapath = os.path.join(PATH_TO_LANDMARK_DATA,r"Train")  # path to save arrays to

    test_imagepath = os.path.join(PATH_TO_DATA,r"Test_Alphabet") # path to load from
    test_datapath = os.path.join(PATH_TO_LANDMARK_DATA,r"Test") # path to save arrays to

    paths = [(train_imagepath,train_datapath),(test_imagepath, test_datapath)]
    data = {}
    for imagepath,datapath in paths:
        data[imagepath] = {}
        skipped = data[imagepath]
        for class_directory in tqdm(os.listdir(imagepath),desc="Classes",position=0): # iterate through every class
            class_path = os.path.join(imagepath,class_directory)
            pathlib.Path(os.path.join(datapath,class_directory)).mkdir(parents=True, exist_ok=True) # if folder does not exist, create it
            skipped[class_directory] = []
            for filename in  tqdm(os.listdir(class_path),desc="Images",position=1,leave=False): # iterate through every image
                file_path = os.path.join(imagepath,class_directory,filename)
                if os.path.isfile(file_path) and file_path.endswith('.png'): # only process png files
                    image = pil.Image.open(file_path)
                    image = landmark_transforms(image)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(image)) #load image to MediaPipe Environment
                    image_array = np.empty((21,2)) #22 landmarks including handedness, 3 coordinates per landmark 
                    if class_directory == "Blank":
                            for i in range(21):
                                image_array[i][0] = 0 # x coordinate of handLandmark
                                image_array[i][1] = 0 # y coordinate of handLandmark
                                # image_array[i][2] = 0 # z coordinate of handLandmark
                            # image_array[21] = 0
                    else:
                        if not image_to_landmarks(image).any():
                            skipped[class_directory].append(filename)
                            continue
                        image_array = image_to_landmarks(image)
                    np.savez(os.path.join(datapath,class_directory,filename),image_array)
    return data[train_imagepath], data[test_imagepath]

def load_landmark_data():
    
    print(f"Loading data from: {PATH_TO_LANDMARK_DATA}")
    train_datapath = os.path.join(PATH_TO_LANDMARK_DATA,r"Train")
    test_datapath = os.path.join(PATH_TO_LANDMARK_DATA,r"Test") 

    full_train_dataset = HandLandmarksDataset(train_datapath)
    train_size = len(full_train_dataset) # compute total size of dataset
    test_dataset = HandLandmarksDataset(test_datapath)

    # Split the datasets into training, validation, and testing sets
    train_dataset, valid_dataset, _ = random_split(full_train_dataset, [int(train_size*0.9),int(train_size*0.1),train_size - int(train_size*0.9)-int(train_size*0.1)])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")
    

    return train_loader,valid_loader,test_loader

def load_device():
    
    # if a macOs then use mps
    if torch.backends.mps.is_built(): device = torch.device("mps")
    
    # elif a GPU is available, use it
    elif torch.cuda.is_available(): device = torch.device("cuda")
    
    # Else, revert to the default (CPU)
    else: device = torch.device("cpu")
        
    return device

def show_images(loader):
    counter=0
    for images, labels in loader:
        if counter == 5 : break
        print(images[counter])
        plt.imshow(images[counter].permute(1, 2, 0))
        plt.title(f"Label: {labels[0]}")
        plt.show()
        counter+=1

if __name__ == "__main__":
    
    download_data()
    # train_loader, valid_loader, test_loader = load_simpleASL_data()
    # show_images(train_loader)
    train_datapath = os.path.join(PATH_TO_LANDMARK_DATA,r"Train")
    test_datapath = os.path.join(PATH_TO_LANDMARK_DATA,r"Test") 
    pathlib.Path(train_datapath).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_datapath).mkdir(parents=True, exist_ok=True)
    train,test = save_landmarks_disk()
    quantified_d = dict((k, len(v)) for k, v in train.items())
    with open(r"results\training\skipped_data.txt", "a") as f:
        f.write(f"{str(quantified_d)}\n")
    for class_name, image_paths in train.items():
        with open(r"results\training\skipped_data.txt", "a") as f:
            f.write(f"{class_name}: ")
            for path in image_paths:
                f.write(f"{path}\n")
            f.write("\n")
    quantified_d = dict((k, len(v)) for k, v in test.items())
    with open(r"results\training\skipped_data.txt", "a") as f:
        f.write(f"{str(quantified_d)}\n")
    for class_name, image_paths in test.items():
        with open(r"results\training\skipped_data.txt", "a") as f:
            f.write(f"{class_name}: ")
            for path in image_paths:
                f.write(f"{path}\n")
            f.write("\n")
    # train_datapath = os.path.join(PATH_TO_LANDMARK_DATA,r"Train")
    # test_datapath = os.path.join(PATH_TO_LANDMARK_DATA,r"Test") 
    # pathlib.Path(train_datapath).mkdir(parents=True, exist_ok=True)
    # pathlib.Path(test_datapath).mkdir(parents=True, exist_ok=True)
    # train,validation,test = load_landmark_data()
    # for X,y in train:
    #     print(X[0])
    
