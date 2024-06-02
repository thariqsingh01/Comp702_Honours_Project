from tkinter import filedialog
import tkinter as tk
import cv2
import numpy as np

# Initialize the Tkinter root
root = tk.Tk()

# Obtain the name of the current directory from the dialog box
directory_path = filedialog.askopenfilename()
image = cv2.imread(directory_path)

root.withdraw()


# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the results of thresholding
edges = cv2.Canny(gray_image, 100, 200)

cv2.imshow('Original Image', image)
cv2.imshow('Threshold of Image', edges)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()