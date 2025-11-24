# Animal-Image-Classifier-Cat-Dog-Fox-Prediction-with-TensorFlow
A complete deep-learning pipeline for classifying animal images into cat, dog, or fox using a TensorFlow/Keras model.
This repository includes a ready-to-run Python script for model inference, batch prediction, evaluation, and image visualization, optimized for VS Code.

📌 Table of Contents

Overview

Key Features

Model Overview

Project Structure

Installation

Configuration

Running the Classifier

Prediction Output Example

Troubleshooting

Future Improvements

License

📘 Overview

This project implements an image classification system that predicts whether an input image contains a:

🐱 Cat

🐶 Dog

🦊 Fox

Using a pre-trained TensorFlow model, the script loads all images in a test directory, preprocesses each one, displays the image, and outputs the predicted label.

The project is structured for clarity, modularity, and easy extension—suitable for students, researchers, and developers working with computer vision.

🚀 Key Features
✔️ 1. Ready-to-Run Prediction Pipeline

Just update your model path and test directory—everything else works automatically.

✔️ 2. Image Preprocessing + Display

Automatic resizing: 224×224×3

Normalization using preprocess_input()

Optional Matplotlib preview of each test image

✔️ 3. Batch Testing

The script evaluates every image in your test folder and produces:

predicted class

total accuracy

filename-based ground truth matching

✔️ 4. Clean & Modular Code

Separate functions for:

loading images

preprocessing

predicting

evaluation

Easy to extend for new datasets.

✔️ 5. VS Code Friendly

No Jupyter dependencies — built to run directly via:

python test_animals.py

🧠 Model Overview

This project expects a TensorFlow/Keras model (e.g., .h5 or SavedModel format) trained on images of cats, dogs, and foxes.
The model must output three logits/probabilities corresponding to:

["cat", "dog", "fox"]


The inference script supports any architecture, including:

MobileNetV2

ResNet50

EfficientNet

Custom CNNs

📁 Project Structure
animal-classifier/
│
├── models/
│   └── animal_model.h5          # Your trained model
│
├── animal_images_dl/
│   └── test/                    # Test images for prediction
│
├── test_animals.py              # Main inference & evaluation script
│
└── README.md                    # Documentation

Supported image formats:
.jpg, .jpeg, .png, .webp

🔧 Installation

Install required packages:

pip install tensorflow matplotlib numpy


(Optional) Add a virtual environment:

python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows

⚙️ Configuration

Inside test_animals.py, modify the following:

MODEL_PATH = "models/animal_model.h5"
TEST_DIR   = "animal_images_dl/test"
CLASSES    = ["cat", "dog", "fox"]

Important:

CLASSES must match the order of your model’s output layer

Images should be placed in the TEST_DIR folder before running

▶️ Running the Classifier

From VS Code terminal or any console:

python test_animals.py


The script will:

Load your trained model

Loop through all images in the test directory

Display each image

Print predicted class

Compute filename-based accuracy

📊 Prediction Output Example
Loading model from: models/animal_model.h5
Running tests on directory: animal_images_dl/test

Prediction for cat_01.jpg: cat
Prediction for dog_33.png: dog
Prediction for wild_fox.webp: fox

Total images: 52
Correct by filename: 47
Accuracy (by filename match): 0.9038

🛠️ Troubleshooting
❗ “Model not found”

Check:

MODEL_PATH = "models/animal_model.h5"


Make sure the file exists.

❗ Prediction results are wrong

Possible causes:

wrong class order

poor model training

images mislabeled

dataset imbalance

❗ "No images found"

Ensure test folder contains .jpg, .jpeg, .png, or .webp.

❗ Model input size mismatch

Your model must accept 224×224×3 or you must change this in the script:

image_utils.load_img(image_path, target_size=(224, 224))

🔮 Future Improvements

Add a confusion matrix

Add confidence scores to output

Create a FastAPI/Flask API for live predictions

Convert model to ONNX or TensorFlow Lite

Add GUI or drag-and-drop web interface

Add training scripts + augmentation
