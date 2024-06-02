# Importing required libraries
from skimage.segmentation import slic
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from skimage import color
from tkinter import filedialog
import tkinter as tk

def enhance_color(image, saturation_scale):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Scale the saturation channel
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255)
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return enhanced_image

def enhance_sharpness(image, kernel_size=3):
    # Define a sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    # Apply the kernel to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Initialize the Tkinter root
root = tk.Tk()

# Obtain the name of the current directory from the dialog box
directory_path = filedialog.askopenfilename()
image = cv2.imread(directory_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
root.withdraw()
# Setting the plot size as 15, 15
plt.figure(figsize=(15, 15))

# Sample Image of scikit-image package


# Applying Simple Linear Iterative
# Clustering on the image
# - 50 segments & compactness = 10
bank_segments = slic(image,
                          n_segments=300,
                          compactness=35)
plt.subplot(1, 2, 1)
print(type(bank_segments))
# Plotting the original image
plt.imshow(image)
plt.subplot(1, 2, 2)
print(type(label2rgb(bank_segments,
                     image,
                     kind='avg')))
# Converts a label image into
# an RGB color image for visualizing
#
toPrint = label2rgb(bank_segments,
                     image,
                     kind='avg')
filtered_image = enhance_sharpness(toPrint)
filtered_image = enhance_color(filtered_image,1.75)
plt.imshow(filtered_image)
plt.show()

import os
from PIL import Image


def transform_and_save_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of image files in the input folder
    image_files = [file for file in os.listdir(input_folder)]

    # Apply transformations and save images to the output folder
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        try:
            with Image.open(image_path) as img:
                # Convert image to grayscale
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                output_image = slic(image,
                          n_segments=400,
                          compactness=30)
                output_image=label2rgb(output_image,
                          image,
                          kind='avg')
                # Save the image
                output_image_path = os.path.join(output_folder, image_file)
                plt.imsave(output_image_path, output_image)
                print(f"Transformed and saved {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


# Define the input and output folder paths
input_folder_path = 'C:\\Users\\Student\\Downloads\\Dataset_preprocess'
output_folder_path = 'C:\\Users\\Student\\Downloads\\Colour_Segmented_Images'

# Apply transformations and save the images
transform_and_save_images(input_folder_path, output_folder_path)