import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
# Load images

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
from PIL import Image
# Initialize the Tkinter root
root = tk.Tk()

# Obtain the name of the current directory from the dialog box
directory_path = filedialog.askopenfilename()
image = cv2.imread(directory_path)

root.withdraw()  # Hide the root window
# Convert the image to grayscale
#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the original and grayscale images
#cv2.imshow('Original Image', image)
#cv2.imshow('Grayscale Image', gray_image)

# Wait indefinitely until a key is pressed
#cv2.waitKey(0)

# Destroy all OpenCV windows
#cv2.destroyAllWindows()

def image_to_pandas(image):
    df = pd.DataFrame([image[:,:,0].flatten(),
                       image[:,:,1].flatten(),
                       image[:,:,2].flatten()]).T
    df.columns = ['Red_Channel','Green_Channel','Blue_Channel']
    return df

df_bank = image_to_pandas(image)

plt.figure(num=None, figsize=(8, 6), dpi=80)
kmeans = KMeans(n_clusters=  3, random_state = 42).fit(df_bank)
result = kmeans.labels_.reshape(image.shape[0],image.shape[1])
print(type(image))
print(type(result))

imshow(result, cmap='viridis')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for n, ax in enumerate(axes.flatten()):
    ax.imshow(result == [n], cmap='gray');
    ax.set_axis_off()

fig.tight_layout()

fig, axes = plt.subplots(1,3, figsize=(15, 12))
for n, ax in enumerate(axes.flatten()):
    image = imread(directory_path)
    image[:, :, 0] = image[:, :, 0]*(result==[n])
    image[:, :, 1] = image[:, :, 1]*(result==[n])
    image[:, :, 2] = image[:, :, 2]*(result==[n])
    ax.imshow(image);
    ax.set_axis_off()
fig.tight_layout()
plt.show()

def pixel_plotter(df):
    x_3d = df['Red_Channel']
    y_3d = df['Green_Channel']
    z_3d = df['Blue_Channel']

    color_list = list(zip(df['Red_Channel'].to_list(),
                          df['Blue_Channel'].to_list(),
                          df['Green_Channel'].to_list()))
    norm = colors.Normalize(vmin=0, vmax=1.)
    norm.autoscale(color_list)
    p_color = norm(color_list).tolist()

    fig = plt.figure(figsize=(12, 10))
    ax_3d = plt.axes(projection='3d')
    ax_3d.scatter3D(xs=x_3d, ys=y_3d, zs=z_3d,
                    c=p_color, alpha=0.55);

    ax_3d.set_xlim3d(0, x_3d.max())
    ax_3d.set_ylim3d(0, y_3d.max())
    ax_3d.set_zlim3d(0, z_3d.max())
    ax_3d.invert_zaxis()

    ax_3d.view_init(-165, 60)
    plt.show()

pixel_plotter(df_bank)

df_bank['cluster'] = result.flatten()


def pixel_plotter_clusters(df):
    x_3d = df['Red_Channel']
    y_3d = df['Green_Channel']
    z_3d = df['Blue_Channel']

    fig = plt.figure(figsize=(12, 10))
    ax_3d = plt.axes(projection='3d')
    ax_3d.scatter3D(xs=x_3d, ys=y_3d, zs=z_3d,
                    c=df['cluster'], alpha=0.55);

    ax_3d.set_xlim3d(0, x_3d.max())
    ax_3d.set_ylim3d(0, y_3d.max())
    ax_3d.set_zlim3d(0, z_3d.max())
    ax_3d.invert_zaxis()

    ax_3d.view_init(-165, 60)


pixel_plotter_clusters(df_bank)

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
                df_bank = image_to_pandas(image)

                plt.figure(num=None, figsize=(8, 6), dpi=80)
                kmeans = KMeans(n_clusters=3, random_state=42).fit(df_bank)
                result = kmeans.labels_.reshape(image.shape[0], image.shape[1])

                # Save the transformed image in the output folder
                # Convert the image to grayscale
                num_clusters = len(np.unique(kmeans.labels_))
                normalized_labels = (result / (num_clusters - 1) * 255).astype(np.uint8)

                # Convert the normalized array to an image
                output_image = Image.fromarray(normalized_labels)
                # Save the image
                output_image_path = os.path.join(output_folder, image_file)
                output_image.save(output_image_path)
                print(f"Transformed and saved {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


# Define the input and output folder paths
input_folder_path = 'C:\\Users\\Student\\Downloads\\Dataset_preprocess'
output_folder_path = 'C:\\Users\\Student\\Downloads\\Segmented Images'

# Apply transformations and save the images
transform_and_save_images(input_folder_path, output_folder_path)