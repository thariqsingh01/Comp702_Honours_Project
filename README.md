# South African Bank Notes Recognition Using Image Processing Techniques

## Project Overview


This project involves the recognition of South African bank notes using advanced image processing techniques. The system can identify and classify old and new bank notes of various denominations (R10, R20, R50, R100, and R200), even when they are presented at different sides, scales, and rotations for accurate classification in financial processes.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Table of Contents


1. Introduction
2. Installation
3. Usage  <br>
 3.1 Image Preprocessing and Enhancement <br>
 3.2 Image Segmentation <br>
 3.3 Feature Extraction <br>
 3.4 Notes Classification <br>
5. Results
6. Conclusion
7. Authors
8. References
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### **1. Introduction:**

   The goal of this project is to create a system that can accurately classify South African bank notes using image preprocessing and enhancement, segmentation, feature extraction, and classification techniques.
   
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### **2. Installation:**
   
   To get started with this project, clone the repository and install the required dependencies or download each file indivdually.<br>
   
   Clone the repository: 
   Bash git clone https://github.com/<your_username>/Comp702_Honours_Project.git <br>
   *Use code with caution. <br>
   
   content_copy Install dependencies: 

   - Original_Dataset.zip
   - Preprocessing & Enhancement.zip
   - ImageSegmentationMethods.zip
   - Feature Extraction.zip
   - CLASSIFICATION.zip
 ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
### **3. Usage:**

 Note: Each script can be run independently or chained together for a complete processing pipeline. Refer to individual script comments for specific execution instructions. Each script contains images that may be used input for the next python project, carefully use the correct image folders as specified in the above files.
The code is well-commented for easy understanding and modification. Experiment with different algorithms and parameters to optimize performance.

  Data Preparation: Place the dataset(Original_Dataset.zip) of bank note images in the data/ directory and add the path of the dataset to the code for retrieval of the bank note images.

   _3.1 Image Preprocessing and Enhancement_ <br>
       Requirements: Install the following libraries: os, glob, cv2, numpy, and matplotlib.pyplot <br>
       Preprocessing: Run the preprocessing script to enhance and normalize the images, you will notice 3 combinations. <br>
       Combination 1: Bilateral Filter, Gamma Correction, Unsharp Masking. <br>
       Combination 2: Gaussian Blur, Histogram Equalization, Sharpening. <br>
       Combination 3: Median Blur, CLAHE, Unsharp Masking. <br>

   _3.2 Image Segmentation_ <br>
       Requirements: Ensure you have various libraries such as OpenCv and sci-kit learn installed, the specific library names can be found in the individual python scripts.<br>
       This project contains 5 python scrips which explores techniques like: Binarization (Otsu's thresholding, etc.) using the Segmentation.py file Canny Edge Segmentation using the Canny_Edge.py file Adaptive Thresholding using the New_Techniques.py file Clustering Algorithms (K-means) using the Techniques_Segmentation.py file Colour Segmentation using the ColorSegmentation.py file <br>
       Input: The above project makes use of the Dataset_preprocess folder to segment images. <br>
       Output: The above project creates a folder Colour_Segmentation which contains all segmented images to be used as input to Feature Extraction. <br>
  
   _3.3 Feature Extraction_ <br>
       Feature Extraction.zip: Extracts key features from segmented images for classification. This script investigates features like:
       Contrast, Dissimilarity,Homogeneity, ASM, Energy and Correlation for 3 different algortithms. These algorithms are GLCM, Haralick and LBP. Since we are working with colour images, the features are 
       extracted for each primary colour(Red, Green and Blue).

Input: The above project uses the folder Colour_Segmentation which is the output from the Image Segmentation Process <br>
Output:The above project will output 3 csv files for each algorithm(GLCM, Haralick and LBP). Each csv file will contain the features for that respective algorithm. The project will also print the features on the terminal(first GLCM, then Haralick and finall LBP), and after will print a comparison of the 3 feature extraction algorithms showing GLCM and Haralick are the better options for feature extraction

Requirements: Open the file feature_extraction.py. Change the paths on lines 72, 141, 214, 354, 356 to the address of your Colour Segmentation folder. Change the paths on lines 75, 144, 21 to the address of where you would like the csv files containing the features to be saved. Lines 345-357 are the addresses of each individual csv file saved from running the algorithm. Once these addresses have been changed, you can successfully run the algorithm and witness the glorius world of feature extraction. Enjoy.

   _3.4 Notes Classification_ <br>
Requirements: Ensure you have the libraries such as os, re, numpy, pandas, and scikit-learn installed. Paths need to be changed before running it, according to the location of your folders.<br>

Description: This project contains a Python script that explores 3 classification techniques using image data<br>
 - Random Forest Classifier<br>
 - K-Nearest Neighbours Classifier<br>
 - Support Vector Machine Classifier<br>

The script uses the Colour_Segmented_Images directory and glcm_features.csv file to segment and extract features from the images. It outputs accuracy scores, classification reports, and confusion matrices for each classifier.<br>

Note: Screenshots of the expected output using 3 different training and test split ratios are available.  

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### **4. Results:**
  
   - The preprocessing step enhances features and reduces noise in the images. <br>
   - Three configurations were tested, and Combination 1 of Bilateral Filter, Gamma Correction, and Unsharp Masking was selected for its balanced performance. <br>
   - Segmentation partitions the images into meaningful regions for further analysis. <br>
   - We tested several techniques and selected colour segmentation for its effectiveness with South African bank notes. <br>
   - Feature extraction converts image data into numerical values that represent important characteristics. <br>
   - We used GLCM, Haralick, and LBP methods, with GLCM and Haralick showing the best performance. <br>
   - We implemented three classifiers: Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM). <br> 
   - SVM was chosen as the most robust classifier based on its high accuracy and consistency. <br>
   - The results showed that the combination of GLCM or Haralick feature extraction with SVM classification provides a reliable solution for bank note recognition. <br>
   
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### **5. Conclusion:**

   This project successfully developed a system for recognizing South African bank notes, demonstrating that advanced image processing techniques combined with effective classification methods can significantly enhance automation in financial applications.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### **6. Authors:**

   - Thariq Singh (219063421@stu.ukzn.ac.za) <br>
   - Pooja Ramahsarai (221003194@stu.ukzn.ac.za) <br>
   - Callyn Blayne Barath (220010761@stu.ukzn.ac.za) <br>
   - Lerisha Moodley (220036955@stu.ukzn.ac.za) <br>

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### **References:**

Please refer to the project documentation for a comprehensive list of references used in this project.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
