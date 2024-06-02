#CLASSIFICATION
import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#EXTRACTING DATA FROM DATASET AND FEATURES
def extract_label(filename):
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def load_image_labels(path):
    labels = []
    file_paths = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            label = extract_label(file)
            if label is not None:
                labels.append(label)
                file_paths.append(file_path)
    return labels, file_paths

def load_features_from_csv(csv_path):
    data = pd.read_csv(csv_path)
    if 'filename' not in data.columns:
        raise ValueError("CSV does not contain 'filename' column")
    data['label'] = data['filename'].apply(lambda x: extract_label(x.split('_')[0]))
    data = data.dropna(subset=['label'])
    data['label'] = data['label'].astype(int)
    X = data.drop(['filename', 'label'], axis=1)
    y = data['label']
    return X, y


dataset_dir = r"C:\\Users\\pooja\\Desktop\\CLASSIFICATION\\Colour_Segmented_Images"
image_labels, file_paths = load_image_labels(dataset_dir)

data = pd.read_csv("C:\\Users\\pooja\\Desktop\\CLASSIFICATION\\glcm_features.csv")
X, y = load_features_from_csv("C:\\Users\\pooja\\Desktop\\CLASSIFICATION\\glcm_features.csv")

#CHECK TO ENSURE LABELS MATCH
if len(image_labels) != len(y):
    print(f"Mismatch: {len(image_labels)} image labels vs {len(y)} CSV labels")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=55)

class_weights = 'balanced'

#RANDOM FOREST CLASSIFIER
print("Training Random Forest Classifier...")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf, zero_division=1))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


#K-NEAREST NEIGHBOUR CLASSIFIER
print("Training K-Nearest Neighbors Classifier...")
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)

print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn)}")
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn, zero_division=1))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


#SUPPORT VECTOR MACHINE CLASSIFIER
print("Training Support Vector Machine Classifier...")
svm_classifier = SVC(kernel='linear', class_weight=class_weights)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm, zero_division=1))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))