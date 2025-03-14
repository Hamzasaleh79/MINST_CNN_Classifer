# MNIST CNN Rotation Predictor

## Overview
This project implements a Convolutional Neural Network (CNN) and an Artificial Neural Network (ANN) using TensorFlow and Keras to predict the rotation angle of handwritten digits from the MNIST dataset. The notebook includes data preprocessing, model training, evaluation, and visualization of results.

## Dataset
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is divided into:
- 60,000 training images
- 10,000 test images

The images are rotated at various angles to create a regression task, where the model predicts the angle of rotation.

## Model Architecture
The project includes two models:
1. **CNN-based Model:**
   - Convolutional layers with ReLU activation
   - Max-pooling layers for feature extraction
   - Fully connected layers for regression output (predicting rotation angles)

2. **ANN-based Model:**
   - Fully connected layers with ReLU activation
   - Regression output layer for angle prediction

## Installation
To run the notebook, install the required dependencies:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Usage
1. Clone this repository:
```bash
git clone https://github.com/your-username/MNIST_CNN_Rotation_Predictor.git
cd MNIST_CNN_Rotation_Predictor
```
2. Open the Jupyter Notebook:
```bash
jupyter notebook CNN_assig.ipynb
```
3. Run the cells to train and evaluate the models.

## Results
The models predict the rotation angle of handwritten digits with high accuracy. The accuracy metric is calculated as:
```python
Accuracy = 100 - (abs(Predicted_Angle - Actual_Angle) / Actual_Angle) * 100
```
Example predictions:
```
Image 1: Predicted angle: 15.2, Actual angle: 15, Accuracy: 98.7%
Image 2: Predicted angle: -10.5, Actual angle: -10, Accuracy: 95%
```

## License
This project is open-source under the MIT License.

