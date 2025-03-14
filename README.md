# MNIST CNN Classifier

## Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The notebook includes data preprocessing, model training, evaluation, and visualization of results.

## Dataset
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is divided into:
- 60,000 training images
- 10,000 test images

## Model Architecture
The CNN model consists of:
- Convolutional layers with ReLU activation
- Max-pooling layers for feature extraction
- Batch Normalization for stable learning
- Fully connected layers for classification

## Installation
To run the notebook, install the required dependencies:
```bash
pip install tensorflow numpy matplotlib opencv-python
