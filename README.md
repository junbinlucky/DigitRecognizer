# Handwritten Digit Recognition Project 
## Project Overview
 This README is for pytrain directory, The repository contains a PyTorch-based implementation for handwritten digit recognition using convolutional neural networks (CNN), with additional model export capabilities.

## File Structure
### Core Implementation Files:
pytorch_mnist.py: Main training script for MNIST digit classification
cnn.py: CNN model architecture definition
### Model Export Utilities:
export_onnx.py: Exports trained model to ONNX format
export_ts.py: Exports trained model to TorchScript format
### Requirements
Python 3.7+
PyTorch 1.8+
torchvision
numpy

## Usage
### Training the Model

> python pytorch_mnist.py

### Exporting Models

#### Export to ONNX
> python export_onnx.py

#### Export to TorchScript
> python export_ts.py

## Performance

The CNN model achieves ~99% accuracy on MNIST test set after 10 epochs of training.

