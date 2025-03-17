# ADAS Road Lane Detection

Deep learning-based lane detection system using convolutional neural networks (CNN) to detect lanes in road videos. The model processes video input, performs lane detection, and outputs a video with detected lane markings overlaid.

## Overview

The project is divided into three main components:
1. **Model**: A neural network (LaneNet) designed for lane detection, built with PyTorch.
2. **Training**: The training script (`train.py`) which trains the model on lane detection data.
3. **Detection**: The detection script (`detect_lanes.py`) that uses the trained model to detect lanes in video footage.
4. **Utility**: Helper functions and datasets for training and detection.

## Demo
timelapse        |  short-demo
:-------------------------:|:-------------------------:
 <img src="https://github.com/user-attachments/assets/d16d5aea-2e96-4cf7-8425-175b0ffd69b1" width="400"> |  <img src="https://github.com/user-attachments/assets/d752bb55-722f-4116-b807-5abffe57856b" width="400">

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- MoviePy
- Scikit-learn
- Numpy

## How the Model Works

The lane detection model is based on a deep learning architecture designed to identify lane markings from road video frames. The model utilizes a convolutional neural network (CNN), which learns to detect lane features from the training data.

### Input:

The model takes video frames as input, each of which is processed to detect lane markings. The frames are typically of a road environment with varying lighting, road types, and lane structures.

### Architecture:

The neural network (LaneNet) is a fully convolutional network (FCN) designed to detect lane boundaries from road images. The architecture includes the following components:

- **Convolutional Layers**: These layers extract spatial features from the input images.
- **Pooling Layers**: These reduce the dimensionality of the image and help in detecting key lane features at multiple scales.
- **Fully Connected Layers**: These layers predict the final output for lane classification.
- **Output Layer**: The output layer produces the lane segmentation mask for each frame, which indicates where lanes are present.

### Training:

The model is trained using images and their corresponding lane segmentation labels. The CNN learns to identify lane markings from the pixel-level ground truth provided in the training dataset.

### Loss Function:

The training process utilizes a **cross-entropy loss function**, which measures the difference between the predicted lane segmentation mask and the actual ground truth. The optimizer adjusts the weights of the network to minimize this loss during training.

### Post-Processing:

After obtaining the lane mask for each frame, post-processing is applied to refine the detected lanes. This may involve filtering out noise, smoothing the lane boundaries, and fitting lane lines to the detected lane pixels.

## Training Process

### Training Data:

For training the model, you need a dataset of road images along with corresponding lane markings. The dataset should contain images from real-world road scenarios with annotated lane markings.
![Figure_1](https://github.com/user-attachments/assets/59383d56-2caa-46ba-8bf3-a6850f20a246)

### Steps for Training:

1. **Prepare Data**: Gather images of roads with lane markings. For each image, create a corresponding label that indicates the lane positions.
2. **Train Model**: Use the training script (`train.py`) to train the model using these images and labels. The script will load the dataset, process the images, and start the training process using the CNN.
3. **Monitor Training**: During training, the model's performance is evaluated using metrics like accuracy or intersection over union (IoU) between the predicted lane mask and ground truth.
4. **Model Saving**: After training, the model weights are saved in a file called `model.pth`, which can be used for inference or testing.

