This GitHub repository contains a Python project for developing an image processing and computer vision system to identify old and new South African banknotes (R10, R20, R50, R100, and R200). The system can handle variations in sides, scales, and rotations for accurate classification in financial applications.

Project Structure:
The project consists of four Python projects:

preprocessing.py: Preprocesses images for better feature extraction. This script allows experimentation with various algorithms:
Grayscale conversion
Noise reduction
Binarization (Otsu's thresholding, etc.)
Image resizing and normalization

ImageSegmentationMethods.zip: Segments images to isolate banknote features. This project contains 5 python scrips which explores techniques like:
Binarization (Otsu's thresholding, etc.) using the Segmentation.py file
Canny Edge Segmentation using the Canny_Edge.py file
Adaptive Thresholding using the New_Techniques.py file
Clustering Algorithms (K-means) using the Techniques_Segmentation.py file
Colour Segmentation using the ColorSegmentation.py file

Input: The above project makes use of the Dataset_preprocess folder to segment images
Output: The above project creates a folder Colour_Segmentation which contains all segmented images to be used as input to Feature Extraction

Requirements: Ensure you have various libraries such as OpenCv and sci-kit learn installed, the specific library names can be found in the individual python scripts

feature_extraction.py: Extracts key features from segmented images for classification. This script investigates features like:
Hu moments (shape)
Color histograms
Texture analysis (Local Binary Patterns)

classification.py: Classifies banknotes based on extracted features. This script explores algorithms like:
Support Vector Machines (SVM)
Convolutional Neural Networks (CNN)
Outputs:

Each script exports processed images for subsequent analysis. The final script outputs the identified banknote denomination.

Usage:
Clone the repository:
Bash
git clone https://github.com/<your_username>/south_african_banknote_identification.git
Use code with caution.
content_copy
Install dependencies:

Refer to individual script comments for specific dependencies (e.g., OpenCV, scikit-image). You can use a package manager like pip for installation.

Run the scripts:

Each script can be run independently or chained together for a complete processing pipeline. Refer to individual script comments for specific execution instructions. Each script contains images that may be used
input for the next python project, carefully use the correct image folders as specified in the above files.

Explore and Modify:

The code is well-commented for easy understanding and modification. Experiment with different algorithms and parameters to optimize performance.
