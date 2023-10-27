import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import cv2
import os
import numpy as np
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import imutils
import time
from os import listdir


def crop_eye_contour(image, plot=False):
    
    #import imutils
    #import cv2
    #from matplotlib import pyplot as plt
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        
        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        plt.title('Original Image')
            
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both', 
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')
        
        plt.show()
    
    return new_image

def get_training_data(classes, training_folder, image_size):
    X = []
    Y = []
    image_width, image_height = image_size

    for cls in classes:
        pth = os.path.join(training_folder, cls)
        for img_file in os.listdir(pth):
            img = cv2.imread(os.path.join(pth, img_file), cv2.IMREAD_COLOR)
            img = crop_eye_contour(img)
            img = cv2.resize(img, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)

            # Convert grayscale image to three channels (RGB)
            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = img / 255

            X.append(img)

             # Create one-hot encoded labels for Y
            if cls.endswith('Moderate'):
                label = [1, 0, 0]
            elif cls.endswith('No_DR'):
                label = [0, 1, 0]
            else:
                label = [0, 0, 1]
            Y.append(label)
            

    X = np.array(X, dtype=np.float32)  # Convert to float32 to reduce memory usage
    Y = np.array(Y, dtype=np.float32)

    # Shuffle the data
    X, Y = shuffle(X, Y, random_state=42)

    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'Y shape is: {Y.shape}')

    return X, Y



# Function to plot sample images
def plot_sample_images(X, y, n=50):
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



def split_data(X, y, test_size=0.3):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)

    return X_train, y_train, X_val, y_val

def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)
    
    score = f1_score(y_true, y_pred)
    
    return score




# Function to format time in hours, minutes, and seconds
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"




# Function to plot training and validation metrics
def plot_metrics(history):
        
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()


"""
# Define your build_model function with 3 convolutional layers
def build_model(input_shape, num_classes):
    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape) # shape=(?, 240, 240, 3)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input) # shape=(?, 244, 244, 3)

    # Convolutional Layer 1
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X) # shape=(?, 238, 238, 32)
    X = Dropout(0.4)(X)

    # Max Pooling Layer 1
    X = MaxPooling2D((4, 4), name='max_pool1')(X) # shape=(?, 59, 59, 32)

    # Convolutional Layer 2
    X = Conv2D(64, (5, 5), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X) # shape=(?, 55, 55, 64)
    X = Dropout(0.4)(X)

    # Max Pooling Layer 2
    X = MaxPooling2D((4, 4), name='max_pool2')(X) # shape=(?, 13, 13, 64)

    # Convolutional Layer 3
    X = Conv2D(128, (3, 3), strides=(1, 1), name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X) # shape=(?, 11, 11, 128)
    X = Dropout(0.4)(X)

    # Flatten Layer
    X = Flatten()(X) # shape=(?, 15488)

    # Fully Connected Layer
    X = Dense(512, activation='relu', name='fc1')(X)
    X = Dropout(0.4)(X)

    # Output Layer
    X = Dense(num_classes, activation='softmax', name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='DR_DetectionModel')

    print(model.summary())
    return model

"""



def build_model(input_shape, num_classes):
    # Load the pre-trained ResNet50 model (without the top layers)

    # Load the ResNet34 model pretrained on ImageNet
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Create a new top layer for the model
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    # Create a new model by combining the base model with the new top layer
    model = Model(inputs=base_model.input, outputs=output)
    
    return model