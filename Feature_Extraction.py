import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import mahotas as mh
from skimage.feature import hog
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,mutual_info_score
from skimage import feature
from skimage.feature import local_binary_pattern
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout


#--------------------------------Gray Level Co-occurrence Matrix(GLCM)---------------------------------------------

# Function to extract GLCM features from an image for each color channel
def extract_glcm_features(image):
    features = {}

    for channel, color_name in enumerate(['red', 'green', 'blue']):
        gray_image_uint8 = img_as_ubyte(image[:, :, channel])

        # Compute the GLCM
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_image_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        ASM = graycoprops(glcm, 'ASM').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()

        features[f'{color_name}_contrast'] = contrast
        features[f'{color_name}_dissimilarity'] = dissimilarity
        features[f'{color_name}_homogeneity'] = homogeneity
        features[f'{color_name}_ASM'] = ASM
        features[f'{color_name}_energy'] = energy
        features[f'{color_name}_correlation'] = correlation
    
    return features

# Function to process each image and extract features
def process_image_glcm(image_path):
    try:
        image = io.imread(image_path)
        if image.ndim == 3 and image.shape[2] == 4:  # Image has 4 channels (including alpha)
            image = image[:, :, :3]  # Discard the alpha channel
        if image.ndim == 3 and image.shape[2] == 3:  # Check if the image has 3 color channels
            features = extract_glcm_features(image)
            original_filename = os.path.basename(image_path)
            features['filename'] = original_filename
            return features
        else:
            print(f"Image {image_path} has {image.shape[2]} channels, expected 3.")
            return None
    
    except Exception as e:
        print(f"Error processing image: {image_path}, Error: {e}")
        return None

# Path to the folder containing images
folder_path = r"D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\Colour_Segmented_Images"

# Path to save the CSV file
save_path = r"D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\FeatureExtraction"

features_list = []

print("GLCM Feature Extraction")

for image_file in os.listdir(folder_path):
    if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(folder_path, image_file)
        features = process_image_glcm(image_path)
        if features:
            features_list.append(features)
            print(f"Image: {features['filename']}")
            for key, value in features.items():
                if key != 'filename':
                    print(f"{key.capitalize()}: {value:.4f}")
            print("\n")

features_df = pd.DataFrame(features_list)
cols = ['filename'] + [col for col in features_df.columns if col != 'filename']
features_df = features_df[cols]
features_df.to_csv(os.path.join(save_path, 'glcm_features.csv'), index=False)

print("Feature extraction and saving completed.")


#-----------------------------------Haralick----------------------------------------


# Function to extract Haralick features from an image for each color channel
def extract_haralick_features(image):
    features = {}

    for channel, color_name in enumerate(['red', 'green', 'blue']):
        gray_image_uint8 = (image[:, :, channel]).astype(np.uint8)
        haralick_features = mh.features.haralick(gray_image_uint8).mean(axis=0)

        features[f'{color_name}_contrast'] = haralick_features[1]
        features[f'{color_name}_dissimilarity'] = haralick_features[4]
        features[f'{color_name}_homogeneity'] = haralick_features[2]
        features[f'{color_name}_ASM'] = haralick_features[0]
        features[f'{color_name}_energy'] = np.sqrt(haralick_features[0])
        features[f'{color_name}_correlation'] = haralick_features[8]
    
    return features

# Function to process each image and extract features
def process_image_haralick(image_path):
    try:
        image = io.imread(image_path)
        if image.ndim == 3 and image.shape[2] == 4:  # Image has 4 channels (including alpha)
            image = image[:, :, :3]  # Discard the alpha channel
        if image.ndim == 3 and image.shape[2] == 3:  # Check if the image has 3 color channels
            features = extract_haralick_features(image)
            original_filename = os.path.basename(image_path)
            features['filename'] = original_filename
            return features
        else:
            print(f"Image {image_path} has {image.shape[2]} channels, expected 3.")
            return None
    
    except Exception as e:
        print(f"Error processing image: {image_path}, Error: {e}")
        return None

# Path to the folder containing images
folder_path = r"D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\Colour_Segmented_Images"

# Path to save the CSV file
save_path = r"D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\FeatureExtraction"

features_list = []

print("Haralick Feature Extraction")

for image_file in os.listdir(folder_path):
    if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(folder_path, image_file)
        features = process_image_haralick(image_path)
        if features:
            features_list.append(features)
            print(f"Image: {features['filename']}")
            for key, value in features.items():
                if key != 'filename':
                    print(f"{key.capitalize()}: {value:.4f}")
            print("\n")

features_df = pd.DataFrame(features_list)

cols = ['filename'] + [col for col in features_df.columns if col != 'filename']
features_df = features_df[cols]

features_df.to_csv(os.path.join(save_path, 'haralick_features.csv'), index=False)

print("Feature extraction and saving completed.")


#----------------------------------Local Binary Pattern(LBP)------------------------------------------------


# Function to extract LBP features from an image for each color channel
def extract_lbp_features(image):
    features = {}

    for channel, color_name in enumerate(['red', 'green', 'blue']):
        gray_image = image[:, :, channel]
        lbp_image = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 10), range=(0, 9))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        for i, hist_value in enumerate(hist):
            features[f'{color_name}_lbp_{i}'] = hist_value
    
    return features

# Function to process each image and extract LBP features
def process_image_lbp(image_path):
    try:
        image = io.imread(image_path)
        if image.ndim == 2:
            # Convert grayscale images to RGB by repeating the channel 3 times
            image = np.stack((image,) * 3, axis=-1)
        if image.ndim == 3 and image.shape[2] == 4:  # Image has 4 channels (including alpha)
            image = image[:, :, :3]  # Discard the alpha channel
        if image.ndim == 3 and image.shape[2] == 3:  # Check if the image has 3 color channels
            features = extract_lbp_features(image)
            original_filename = os.path.basename(image_path)
            features['filename'] = original_filename
            return features
        else:
            print(f"Image {image_path} has {image.shape[2]} channels, expected 3.")
            return None
    
    except Exception as e:
        print(f"Error processing image: {image_path}, Error: {e}")
        return None

# Path to the folder containing images
folder_path = r"D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\Colour_Segmented_Images"

# Path to save the CSV file
save_path = r"D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\FeatureExtraction"

features_list = []

print("LBP Feature Extraction")

for image_file in os.listdir(folder_path):
    if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(folder_path, image_file)
        features = process_image_lbp(image_path)
        if features:
            features_list.append(features)
            print(f"Image: {features['filename']}")
            for key, value in features.items():
                if key != 'filename':
                    print(f"{key.capitalize()}: {value:.4f}")
            print("\n")

features_df = pd.DataFrame(features_list)

cols = ['filename'] + [col for col in features_df.columns if col != 'filename']
features_df = features_df[cols]

features_df.to_csv(os.path.join(save_path, 'lbp_features.csv'), index=False)

print("Feature extraction and saving completed.")


#------------------------------Compare the 3 algorithms - Feed Forward Neural Network---------------------------------------


label_encoder = LabelEncoder()

# Function to extract LBP features from each color channel of an image
def extract_lbp_features(image):
    features = {}

    for channel, color_name in enumerate(['red', 'green', 'blue']):
        gray_image = image[:, :, channel]
        lbp_features = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, 10), range=(0, 9))
        lbp_features_normalized = hist.astype(np.float64) / hist.sum()
        
        contrast = np.std(lbp_features_normalized)
        dissimilarity = np.mean(np.abs(np.diff(lbp_features_normalized)))
        homogeneity = np.mean(1 / (1 + np.square(lbp_features_normalized)))
        ASM = np.mean(np.square(lbp_features_normalized))
        energy = np.mean(np.abs(lbp_features_normalized))
        correlation = np.mean(lbp_features_normalized * np.arange(0, len(lbp_features_normalized)) - np.mean(lbp_features_normalized))

        features[f'{color_name}_contrast'] = contrast
        features[f'{color_name}_dissimilarity'] = dissimilarity
        features[f'{color_name}_homogeneity'] = homogeneity
        features[f'{color_name}_ASM'] = ASM
        features[f'{color_name}_energy'] = energy
        features[f'{color_name}_correlation'] = correlation
    
    return features

# Function to process each image and extract features
def process_image_lbp(image_path):
    try:
        image = io.imread(image_path)
        if image.shape[2] != 3:
            raise ValueError(f"Image {image_path} does not have 3 color channels.")
        features = extract_lbp_features(image)
        original_filename = os.path.basename(image_path)
        features['filename'] = original_filename
        return features
    
    except Exception as e:
        print(f"Error processing image: {image_path}, Error: {e}")
        return None

# Function to load and preprocess features from CSV file
def load_and_preprocess_features(feature_file):
    features = pd.read_csv(feature_file)
    X = features.drop(columns=['filename']) 
    y = features['filename'] 
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    sel = VarianceThreshold(threshold=0.1)
    X_train_selected = sel.fit_transform(X_train_normalized)
    X_test_selected = sel.transform(X_test_normalized)
    return X_train_selected, X_test_selected, y_train, y_test, label_encoder.classes_

def perturb_image(image, noise_level=0.1):
    noisy_image = image + np.random.normal(loc=0.0, scale=noise_level, size=image.shape)
    return noisy_image

def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(X_train, y_train, input_shape, num_classes):
    model = create_model(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    return model

def compute_texture_complexity(contrast, dissimilarity, ASM, energy):
    texture_complexity = (contrast + dissimilarity + ASM + energy) / 4
    return texture_complexity

def compute_overall_image_quality(contrast, homogeneity, energy):
    overall_quality = (contrast + homogeneity + energy) / 3
    return overall_quality

def compute_pattern_recognition(correlation, homogeneity):
    pattern_metric = (correlation + homogeneity) / 2
    return pattern_metric

def compute_anomaly_detection(contrast, dissimilarity):
    anomaly_metric = (contrast + dissimilarity) / 2
    return anomaly_metric

# Load and preprocess features for GLCM, Haralick, and LBP
X_train_glcm, X_test_glcm, y_train_glcm, y_test_glcm, classes_glcm = load_and_preprocess_features("D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\FeatureExtraction\\glcm_features.csv")
X_train_haralick, X_test_haralick, y_train_haralick, y_test_haralick, classes_haralick = load_and_preprocess_features("D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\FeatureExtraction\\haralick_features.csv")
X_train_lbp, X_test_lbp, y_train_lbp, y_test_lbp, classes_lbp = load_and_preprocess_features("D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\FeatureExtraction\\lbp_features.csv")

# Collect features from images for additional computations
all_features = []
lbp_homogeneity_list = []
lbp_correlation_list = []

for image_file in os.listdir("D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\Colour_Segmented_Images"):
    if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join("D:\\Varsity\\Honours\\Semester 1\\COMP702\\Mini Project\\GroupProjectComp702\\Assign\\Colour_Segmented_Images", image_file)

        image = io.imread(image_path)
        features = extract_lbp_features(image)
        lbp_homogeneity_list.append(features['red_homogeneity'])
        lbp_correlation_list.append(features['red_correlation'])

# Compute mean values for LBP homogeneity and correlation
mean_lbp_homogeneity = np.mean(lbp_homogeneity_list)
mean_lbp_correlation = np.mean(lbp_correlation_list)

# Create DataFrame for all features
features_df = pd.DataFrame(all_features)

# GLCM features extraction
glcm = graycomatrix(image[:, :, 0], distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
glcm_contrast = graycoprops(glcm, 'contrast')
glcm_dissimilarity = graycoprops(glcm, 'dissimilarity')
glcm_ASM = graycoprops(glcm, 'ASM')
glcm_energy = graycoprops(glcm, 'energy')
glcm_homogeneity = graycoprops(glcm, 'homogeneity')
glcm_correlation = graycoprops(glcm, 'correlation')

# Haralick features extraction
haralick_contrast = graycoprops(glcm, 'contrast')
haralick_dissimilarity = graycoprops(glcm, 'dissimilarity')
haralick_ASM = graycoprops(glcm, 'ASM')
haralick_energy = graycoprops(glcm, 'energy')
haralick_homogeneity = graycoprops(glcm, 'homogeneity')
haralick_correlation = graycoprops(glcm, 'correlation')

# LBP features extraction
lbp_features = local_binary_pattern(image[:, :, 0], P=8, R=1, method='uniform')
hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, 10), range=(0, 9))
lbp_contrast = np.std(hist)
lbp_dissimilarity = np.mean(np.abs(np.diff(hist)))
lbp_ASM = np.mean(np.square(hist))
lbp_energy = np.mean(np.abs(hist))

# Compute GLCM, Haralick, and LBP metrics
contrast_glcm = np.mean(glcm_contrast)  
dissimilarity_glcm = np.mean(glcm_dissimilarity) 
ASM_glcm = np.mean(glcm_ASM)  
energy_glcm = np.mean(glcm_energy) 
homogeneity_glcm = np.mean(glcm_homogeneity)  
correlation_glcm = np.mean(glcm_correlation)  

contrast_haralick = np.mean(haralick_contrast)  
dissimilarity_haralick = np.mean(haralick_dissimilarity) 
ASM_haralick = np.mean(haralick_ASM)  
energy_haralick = np.mean(haralick_energy)  
homogeneity_haralick = np.mean(haralick_homogeneity)  
correlation_haralick = np.mean(haralick_correlation) 

contrast_lbp = np.mean(lbp_contrast)  
dissimilarity_lbp = np.mean(lbp_dissimilarity) 
ASM_lbp = np.mean(lbp_ASM)  
energy_lbp = np.mean(lbp_energy)  
homogeneity_lbp = np.mean(features['red_homogeneity']) 
correlation_lbp = np.mean(features['red_correlation'])  

# Compute texture complexity, overall quality, pattern recognition, and anomaly detection metrics
texture_complexity_glcm = compute_texture_complexity(contrast_glcm, dissimilarity_glcm, ASM_glcm, energy_glcm)
texture_complexity_haralick = compute_texture_complexity(contrast_haralick, dissimilarity_haralick, ASM_haralick, energy_haralick)
texture_complexity_lbp = compute_texture_complexity(contrast_lbp, dissimilarity_lbp, ASM_lbp, energy_lbp)

overall_quality_glcm = compute_overall_image_quality(contrast_glcm, homogeneity_glcm, energy_glcm)
overall_quality_haralick = compute_overall_image_quality(contrast_haralick, homogeneity_haralick, energy_haralick)
overall_quality_lbp = compute_overall_image_quality(contrast_lbp, homogeneity_lbp, energy_lbp)

pattern_metric_glcm = compute_pattern_recognition(correlation_glcm, homogeneity_glcm)
pattern_metric_haralick = compute_pattern_recognition(correlation_haralick, homogeneity_haralick)
pattern_metric_lbp = compute_pattern_recognition(correlation_lbp, homogeneity_lbp)

anomaly_metric_glcm = compute_anomaly_detection(contrast_glcm, dissimilarity_glcm)
anomaly_metric_haralick = compute_anomaly_detection(contrast_haralick, dissimilarity_haralick)
anomaly_metric_lbp = compute_anomaly_detection(contrast_lbp, dissimilarity_lbp)

# Print metrics
print("GLCM Texture Complexity:", texture_complexity_glcm)
print("GLCM Overall Image Quality:", overall_quality_glcm)
print("GLCM Pattern Recognition Metric:", pattern_metric_glcm)
print("GLCM Anomaly Detection Metric:", anomaly_metric_glcm)

print("Haralick Texture Complexity:", texture_complexity_haralick)
print("Haralick Overall Image Quality:", overall_quality_haralick)
print("Haralick Pattern Recognition Metric:", pattern_metric_haralick)
print("Haralick Anomaly Detection Metric:", anomaly_metric_haralick)

print("LBP Texture Complexity:", texture_complexity_lbp)
print("LBP Overall Image Quality:", overall_quality_lbp)
print("LBP Pattern Recognition Metric:", pattern_metric_lbp)
print("LBP Anomaly Detection Metric:", anomaly_metric_lbp)

# Print mean LBP homogeneity and correlation
print("Mean LBP Homogeneity:", mean_lbp_homogeneity)
print("Mean LBP Correlation:", mean_lbp_correlation)

