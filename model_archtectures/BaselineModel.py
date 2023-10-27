import os
import cv2
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  # Import the Random Forest classifier

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, precision_score, recall_score

class BaselineModel:
    def __init__(self):
        self.model = None

    def get_training_data(self, directory_path, classes):
        # Construct the path to the provided directory
        X = []
        Y = []
        for cls in classes:
            pth = directory_path + cls    
            for j in os.listdir(pth):
                img = cv2.imread(pth+'/'+j, 0)
                img = cv2.resize(img, (200,200))
                X.append(img)
                Y.append(classes[cls])

        # Convert image data and labels to NumPy arrays
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    
    # Function to plot sample images
    def plot_sample_images(self, X, y, n=50):
        for label in [0, 1]:
            images = X[np.argwhere(y == label)]
            n_images = images[:n]
            columns_n = 10
            rows_n = int(n / columns_n)
            plt.figure(figsize=(20, 10))
            i = 1
            for image in n_images:
                plt.subplot(rows_n, columns_n, i)
                plt.imshow(image[0], cmap='gray' if len(image[0].shape) == 2 else None)
                plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False,
                                labelbottom=False, labeltop=False, labelleft=False, labelright=False)
                i += 1
            label_to_str = lambda label: "No" if label == 1 else "Moderate"
            plt.suptitle(f"DR: {label_to_str(label)}")
            plt.show()

    def display_data_info(self, X, Y):
        # Display the unique class labels
        unique_y = np.unique(Y)
        print("Unique classes:", unique_y)

        # Display the count of images per class
        class_counts = pd.Series(Y).value_counts()
        print("Class counts:\n", class_counts)

        # Display the shape of the image data array
        print("X shape:", X.shape)

    
    def preprocess_data(self, X):
        # Reshape the image data array to be a 2D array (prepare data)
        X_updated = X.reshape(len(X), -1)
        print("X updated shape:", X_updated.shape)

        # Scale the features
        X_updated = X_updated / 255.0
        return X_updated

    def split_data(self, X, Y):
        # Split the dataset into training and validation sets
        xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, random_state=10, test_size=0.20)
        print("Training set shape:", xtrain.shape)
        print("Testing set shape:", xvalid.shape)
        return xtrain, xvalid, ytrain, yvalid

    def train_and_evaluate_svc_model(self, xtrain, ytrain, xvalid, yvalid):
        # Train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(xtrain, ytrain)

        # Evaluate the model on the validation data
        y_pred = self.model.predict(xvalid)
        accuracy = accuracy_score(yvalid, y_pred)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

        
        # Save the trained model to a file using pickle
        model_filename = "random_forest_model.pkl"
        with open(model_filename, "wb") as model_file:
            pickle.dump(self.model, model_file)

        print(f"Model saved to {model_filename}")

    def view_metrics(self, x_test, y_test):
        y_pred = self.model.predict(x_test)

        class_report = classification_report(y_test, y_pred, target_names=["No_DR", "Moderate", "Proliferate_DR"])

        print("Classification Report:")
        print(class_report)

        # Calculate accuracy for each class
        accuracy_per_class = accuracy_score(y_test, y_pred, normalize=False)
        total_samples_per_class = [np.sum(y_test == i) for i in range(len(np.unique(y_test)))]
        
        for i, class_name in enumerate(["No_DR", "Moderate", "Severe"]):
            class_accuracy = accuracy_per_class[i] / total_samples_per_class[i]
            print(f"Accuracy for {class_name}: {class_accuracy * 100:.2f}%")


        # Calculate precision, recall, and F1 score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        

