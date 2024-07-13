# Speech Emotion Recognition

This project focuses on recognizing emotions from a person's speech using machine learning and deep learning techniques. The primary objective is to process human speech signals and display the corresponding emotion. 

## Overview

Speech Emotion Recognition (SER) is essential for improving human-computer interaction and has significant applications in healthcare. This project leverages the RAVDESS dataset for training models to identify different emotions such as Happy, Sad, Angry, etc. The Librosa package is used for feature extraction from the speech data.

## Dataset

- **RAVDESS Dataset**: The dataset includes recordings from 24 trained actors (12 male, 12 female) who spoke two lexically-matched statements in a neutral North American accent. Emotions include calm, happy, sad, angry, fearful, disgust, surprised, and neutral. 

## Data Preprocessing

- **Data Augmentation**: Techniques like adding noise, shifting, stretching, and pitch adjustment are applied to increase the diversity of the training data.
- **Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients), ZCR (Zero Crossing Rate), and RMSE (Root Mean Square Energy) features are extracted.
- **Standardization**: Features are standardized to have a mean of 0 and a standard deviation of 1.
- **Encoding**: One-hot encoding is used for the categorical emotion labels.
- **Splitting Data**: Data is split into training (80%) and testing (20%) sets.

## Models

1. **Decision Tree**
2. **K-Nearest Neighbors (KNN)**
3. **Convolutional Neural Network (CNN)**

### CNN Model Architecture

- **Input Layer**: 1D Convolutional layer with 64 filters.
- **Pooling Layer**: MaxPooling1D layer to reduce dimensionality.
- **Flattening Layer**: Flatten layer to convert the 2D matrix data to a vector.
- **Dense Layers**: Fully connected layers with ReLU activation and Dropout regularization.
- **Output Layer**: Dense layer with softmax activation for multi-class classification.

### Evaluation

- **Accuracy**: The CNN model achieved an accuracy of approximately 97.27% on the training set and 72.5% on the test set.
- **Confusion Matrix**: Used to visualize the performance of the classification model.

## Installation

### Prerequisites

- Python 3.7 or higher

### Required Libraries

Install the required libraries using the following command:

```bash
pip install -r requirements.txt
