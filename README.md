# Project Name: **MobileNetV2 Model Training on Custom Image Dataset**

## Description

This project demonstrates how to fine-tune and train the MobileNetV2 model on a custom image dataset for binary classification tasks using TensorFlow and Keras. The model architecture is based on MobileNetV2, a lightweight and efficient convolutional neural network designed for mobile and edge devices.

The dataset is divided into three subsets:
- **70% for training**
- **10% for validation**
- **20% for testing**

These proportions can be adjusted based on specific needs.

## Features

- **Data Loading and Preprocessing**: Images are preprocessed using the `ImageDataGenerator` with the MobileNetV2 preprocessing function to ensure optimal performance.
- **Model Architecture**: MobileNetV2 is used as the base model, with additional layers added for binary classification.
- **Training**: The model is trained with early stopping and model checkpointing to avoid overfitting and ensure the best model is saved.
- **Evaluation**: Includes accuracy and loss visualization, as well as a confusion matrix and classification report.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Google Colab (if using the notebook in a cloud environment)

## Run the code:

If you are using Google Colab, you can upload the script and run it directly in a notebook.

For running locally, ensure that the paths to your datasets are correctly set up in the code. The dataset paths should be relative to the project directory to avoid errors.

Modify the dataset split ratios or paths as needed to suit your specific project requirements.

## Dataset Preparation
The dataset used in this project is expected to be organized into separate directories for training, validation, and testing. Ensure that your image dataset follows this structure:
- Training_Dataset/
  - Class1/
  - Class2/
- Validation_Dataset/
  - Class1/
  - Class2/
- Test_Dataset/
  - Class1/
  - Class2/
