# Edge-AI-Face-Recognition
An edge-optimized face recognition pipeline for the Jetson Orin Nano, featuring a CNN-based 'Smart Gating' mechanism for dynamic image restoration in non-ideal environments.

How the Project Works (System Pipeline)
The system operates as a real-time inference pipeline optimized for edge hardware. The process consists of 5 main stages:

Acquisition: The system receives a live video stream via a USB or CSI camera connected to the Jetson Orin Nano.

Detection: A TensorRT-optimized YOLOv8 model performs real-time face detection and extracts bounding box crops.

Smart Gating (Quality Classification): The face crop passes through a lightweight Convolutional Neural Network (SimpleGateCNN). This network classifies the image into one of four categories: Normal, Low-Light, Motion-Blur, or Low-Resolution.

Restoration (Dynamic Routing): Based on the classification, the image is dynamically routed to a specific restoration agent:

CLAHE: Enhances visibility in low-light conditions.

Laplacian Sharpening: Applies edge enhancement to correct motion blur.

ESPCN / FSRCNN: Upscales low-resolution images.

Identification: The normalized or restored face crop is fed into a ResNet-18 model for final identity verification.

Repository Structure & File Descriptions
folder excecution_files:
1)main.py
The central entry point of the application. It orchestrates the asynchronous pipeline, manages      camera I/O, executes the gating logic, and controls data flow between the detection,                restoration, and identification models.

2)benchmark.py
The evaluation and testing suite. It runs comparative analysis between a static baseline            pipeline and the adaptive gating pipeline, generating metrics for Accuracy, Throughput (FPS),       and End-to-End Latency.

folder models:

1)gate.py
Defines the architecture for the classification network (the Gate). This is the core of the "Mixture of Experts" mechanism, ensuring compute resources are saved by only running heavy restoration algorithms when strictly necessary.

2)yoloV8.py
Handles loading the YOLOv8 model, executing GPU inference, and extracting precise face crops from the video feed.

3)ResNet.py
Manages the final facial recognition stage. It receives the processed face crop and computes the feature embeddings to output the recognized identity.

folder restoration_agents 
Houses the logic for all image processing and restoration algorithms (CLAHE, Laplacian, ESPCN). Functions here leverage CUDA-accelerated OpenCV to maintain ultra-low latency.

folder train
Files used to train the built models - the gate and resnet

folder genarate_data
Contains the files for processing images and creating datasets and tests for the project.
