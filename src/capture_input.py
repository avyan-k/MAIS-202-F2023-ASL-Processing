import cv2
import os
import time

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

        # Save the image with a unique filename
        image_filename = os.path.join(save_dir, f"image_{i}.png")
        cv2.imwrite(image_filename, frame)

        print(f"Image {i+1} captured and saved as {image_filename}")

        # Wait for 1 second
        cv2.waitKey(1000)
        i+=1

    # Release the webcam and close OpenCV
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()