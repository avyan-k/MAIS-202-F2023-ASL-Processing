import cv2
import numpy as np


def processImage(path : str)->list[list[int]]:
    '''
    loads and processes image for convolution
    '''
    loaded_image = cv2.imread(path) #load image from path
    grayscale = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2GRAY) #converts image to grayscale, can be commentted out
    return grayscale

def convolutionImage(image : np.array, kernel : np.array) -> list[list[int]]: 
    '''
    assume no padding or strides for now
    takes an input image and filter kernel and applies convolution
    '''
    image_width,image_height = image.shape #extract dimensions of input image
    kernel_width, kernel_height = kernel.shape #extract dimensions of filter kernel
    convoluted_image = np.zeros((image_height-kernel_height +1,image_width-kernel_width+1)) #set dimensions of returned image
    for y in range(convoluted_image.shape[1]-1):
        for x in range(convoluted_image.shape[0]-1):
            convoluted_image[x,y] = np.dot(image[x:x+kernel_width,y:y+kernel_height].flatten(),kernel.flatten())  #convolution formula
    return convoluted_image





if __name__ == "__main__":
    test1 = processImage(r"images/test_image1.png")
    test2 = processImage(r"images/test_image2.png")
    #sobel operators, taken from https://www.tutorialspoint.com/dip/sobel_operator.htm
    sobel_x = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    sobel_y = np.rot90(sobel_x,-1)
    convolution1 = convolutionImage(convolutionImage(test1,sobel_y),sobel_x) #applies sobel horizontally and vertically for edge detection
    convolution2 = convolutionImage(convolutionImage(test2,sobel_x),sobel_y)
    cv2.imshow('Image1',convolution1)
    cv2.imshow('Image2',convolution2)
    cv2.waitKey()