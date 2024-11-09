
# Face Detection & Recognition on Pins FR Dataset using CNNs & Transfer Learning

    This project builds a Face Recognition System using deep learning to detect and identify faces in images or video. The system uses the VGG16 model fine-tuned for face recognition tasks. It recognizes faces of celebrities from a pre-trained dataset and can also identify new faces by comparing their features.

Key features of the system:

    Face Detection: It detects faces in images or video using OpenCV's deep learning-based model.

    Face Recognition: The VGG16 model generates unique facial embeddings, allowing it to identify known celebrities.

    Cosine Similarity: For new faces, the system compares facial features with stored ones and labels the face as either a recognized celebrity or "not recognized."

    Real-Time Recognition: It works with live camera feeds for real-time face recognition.

The project also tracks the model's performance with accuracy and loss metrics, visualized over time.



# Dataset description

    The dataset used in this project contains images of various celebrities, which are used for training and fine-tuning the face recognition model. Each celebrity has a collection of images with different poses, lighting, and expressions to help the model learn distinctive facial features.

Key details about the dataset:

    Total Number of Classes: 105 unique celebrity identities.

    Images per Class: Each celebrity has multiple images, with variations in facial expressions and angles.

    Image Resolution: Images are resized to 224x224 pixels to match the input size of the VGG16 model.

    Labels: Each image is labeled with the name of the corresponding celebrity.

This dataset is used to train the model to recognize and classify faces by learning unique facial embeddings. After training, the model can identify faces of celebrities that it has been exposed to, as well as handle faces that were not in the original training set by using cosine similarity.



## Steps to Run the Code in Jupyter Notebook

Steps to Run the Code in Jupyter Notebook

Load the Dataset

    Import Necessary Libraries: First, you need to import the required libraries such as tensorflow, numpy, and cv2.

    Load the Dataset: Load your dataset (for example, the Fashion MNIST dataset or your custom dataset) into the notebook.
    Ensure the dataset is correctly loaded and split into training and testing sets.

Preprocess the Data

    Normalize the Data: Normalize the images by scaling the pixel values between 0 and 1 to ensure the model can train effectively.

    Flatten the Images: For certain types of models, such as fully connected networks, you'll need to flatten the images into one-dimensional vectors.

    For CNN-based models (like VGG16), resizing the images to the correct dimensions (e.g., 224x224) will be necessary.

    Visualize the Data: It is a good idea to visualize some images from the dataset to ensure the preprocessing steps are correct.

Build the Neural Network Model

    Create the Model: Build your neural network model using tensorflow.keras.Sequential. If using transfer learning, load a pre-trained model like VGG16, and add custom layers on top.

    Include layers such as flatten, dense (with ReLU activation), and dropout.

    Add a final dense layer for classification with a softmax activation function.

    Compile the Model: Choose an optimizer (e.g., Adam), specify the loss function (sparse_categorical_crossentropy for multi-class classification), and define the evaluation metrics (e.g., accuracy).

Train the Model

    Train on the Dataset: Fit the model using the training data, typically using an 80-20 split for training and validation.

    Monitor the loss and accuracy on the validation data to evaluate the model’s performance.

    Early Stopping & Learning Rate Scheduler: To prevent overfitting and enhance training, implement early stopping and a learning rate scheduler.

Fine-Tune the Model

    Unfreeze Some Layers: Once the initial training is complete, unfreeze the last few layers of the pre-trained model.

    Lower the Learning Rate: When fine-tuning, it’s essential to use a lower learning rate to avoid drastically changing the pre-trained features.

    Retrain the Model: Continue training with both the new layers and the unfreezed base layers to improve the model's performance.

Save the Model

    Save the Model: Once training and fine-tuning are complete, save the model using model.save() for future use or deployment.    

Evaluate the Model

    Test Accuracy: After training and fine-tuning, evaluate the model using a test dataset that was not used during training.

    Confusion Matrix: Visualize the confusion matrix to understand class-wise performance and identify any misclassifications.

    Classification Report: Generate a classification report showing precision, recall, and F1-scores for each class.

Visualize the Results

    Plot Loss Curves: Plot the training and validation loss curves to check for any overfitting or underfitting.

    Plot Accuracy Curves: Similarly, plot the accuracy curves to observe how well the model is performing during training and validation.

    Precision, Recall, and F1-Score Bar Graph: Display a bar graph for precision, recall, and F1-score of each class for better understanding of model performance.


## Dependencies and Installation Instructions
import cv2

import os

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import VGG16

from tensorflow.keras.models import Model

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, 
Dropout, BatchNormalization, LeakyReLU

from tensorflow.keras.layers import Dense, Flatten, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, 
EarlyStopping

from tensorflow.keras.regularizers import l2

from tensorflow.keras.applications import InceptionV3

import tensorflow as tf

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity

import pickle

from IPython.display import display, clear_output

from PIL import Image

import os
