# Q5 - Virtual Try-On Evaluation System

## Overview

This project implements a modular Virtual Try-On evaluation pipeline.

The system takes a person image and garment image as input and processes them through different computer vision modules to generate and evaluate a virtual try-on result.

The implementation is designed with separate modules for detection, pose estimation, try-on generation, and evaluation.

---

## Features

- Person and garment image processing
- Object detection module
- Human pose analysis
- Virtual try-on pipeline
- Result evaluation
- Streamlit-based user interface
- Modular architecture for easy extension

---

## System Pipeline
          Input Images

    Person Image + Garment Image
                |
                v

    +---------------------+
    |  Detection Module   |
    |  Image Analysis     |
    +---------------------+

                |
                v

    +---------------------+
    |   Pose Module       |
    | Human Body Analysis |
    +---------------------+

                |
                v

    +---------------------+
    |   Try-On Module     |
    | Garment Transfer    |
    +---------------------+

                |
                v

    +---------------------+
    | Evaluation Module   |
    | Quality Assessment  |
    +---------------------+

                |
                v

      Virtual Try-On Result.   


      
---

## Module Description

### detector.py

Handles image detection and preprocessing operations required for the try-on pipeline.

### pose.py

Responsible for extracting human pose information and body-related features.

### tryon.py

Contains the main virtual try-on processing logic that combines the person and garment information.

### evaluation.py

Provides evaluation utilities to analyze the generated output quality.

### utils.py

Contains helper functions shared across different modules.

---

## Technology Stack

- Python
- Computer Vision
- Deep Learning
- PyTorch
- OpenCV
- Streamlit

---

## Running the Application

### Install Dependencies

```bash
pip install -r requirements.txt
