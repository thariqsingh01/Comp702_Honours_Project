import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk

# Initialize the Tkinter root
root = tk.Tk()

# Obtain the name of the current directory from the dialog box
directory_path = filedialog.askopenfilename()
image = cv2.imread(directory_path)

root.withdraw()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(type(gray_image))
# Display the results of thresholding
_, binary_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('Original Image', image)
cv2.imshow('Threshold of Image', binary_otsu)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()