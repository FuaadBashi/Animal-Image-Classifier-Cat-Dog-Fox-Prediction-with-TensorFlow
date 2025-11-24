# Animal Image Classifier 🐱🐶🦊

A simple and modular workflow for running predictions on animal images using a pre-trained deep learning model. This repository provides an easy-to-use TensorFlow inference pipeline for image classification.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Description](#model-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Example Output](#example-output)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [License](#license)

## 🎯 Overview

This project provides a ready-to-use prediction system that loads images from a directory, preprocesses them to match the model's input format, and outputs predicted labels. A filename-based validation method is included for quick accuracy evaluation.

**Ideal for:** Students, researchers, and developers working with image classification and TensorFlow inference pipelines.

## ✨ Features

### 1. **Ready-to-Use Prediction Script**
- Single Python script (`test_animals.py`) handles the entire inference pipeline
- Automatic model loading, preprocessing, prediction, and accuracy computation

### 2. **Automatic Image Preprocessing**
- Resizes images to 224×224×3
- Converts to arrays
- Normalizes inputs using `preprocess_input`

### 3. **Batch Image Evaluation**
- Runs predictions on all `.jpg`, `.jpeg`, `.png`, and `.webp` files in the test directory

### 4. **Optional Image Display**
- Uses Matplotlib to preview each image before running prediction

### 5. **Configurable Class Mapping**
- Class names are manually defined to match your model's output layer

## 🧠 Model Description

The classifier uses a TensorFlow/Keras model trained to recognize three categories:

- **cat** 🐱
- **dog** 🐶
- **fox** 🦊

The script supports any model saved in Keras `.h5` or TensorFlow `SavedModel` format. The only requirement is that the model outputs three probabilities in the same order as the `CLASSES` list defined in the script.

## 📁 Project Structure
```
animal-classifier/
│
├── models/
│   └── animal_model.h5              # Your trained model
│
├── animal_images_dl/
│   └── test/                        # Images for prediction
│       ├── cat_01.jpg
│       ├── dog_02.png
│       └── fox_03.webp
│
├── test_animals.py                  # Main Python script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

**Supported image formats:**
- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/animal-classifier.git
cd animal-classifier
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install tensorflow matplotlib numpy
```

### Step 3: (Optional) Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

## ⚙️ Configuration

Open `test_animals.py` and update the following section:
```python
MODEL_PATH = "models/animal_model.h5"
TEST_DIR   = "animal_images_dl/test"
CLASSES    = ["cat", "dog", "fox"]
```

**Ensure:**
1. ✅ The model path is correct
2. ✅ Your test directory contains images
3. ✅ The class list matches the model's output order

## 💻 Usage

Run the script from your terminal:
```bash
python test_animals.py
```

The script will:
1. Load the model
2. Scan the test directory
3. Display each image (optional)
4. Predict its class
5. Compare prediction to filename
6. Print overall accuracy

## 📊 Example Output
```
Loading model from: models/animal_model.h5
Running tests on directory: animal_images_dl/test

Prediction for cat_02.jpg: cat
Prediction for dog_15.png: dog
Prediction for fox_08.webp: fox

Total images: 52
Correct by filename: 47
Accuracy (by filename match): 0.9038
```

## 🔧 Troubleshooting

### "Model not found"
**Solution:** Check that `MODEL_PATH` correctly points to your model file.

### "No images found"
**Solution:** Ensure `TEST_DIR` contains supported image files (`.jpg`, `.jpeg`, `.png`, `.webp`).

### Incorrect predictions
**Possible causes:**
- Verify class order in `CLASSES`
- Review model training quality
- Check for mislabeled filenames
- Ensure consistent image sizes/resolution

### Input size mismatch
If your model expects a different input size, modify this line:
```python
image_utils.load_img(image_path, target_size=(224, 224))
```

Change `(224, 224)` to match your model's expected input dimensions.

## 🚧 Future Work

- [ ] Add precision, recall, and confusion matrix
- [ ] Display confidence scores for predictions
- [ ] Convert model to ONNX or TFLite
- [ ] Implement a FastAPI/Flask API endpoint
- [ ] Add drag-and-drop web interface
- [ ] Provide official training notebooks
- [ ] Implement test-time augmentation
- [ ] Support for additional animal classes
- [ ] Real-time webcam prediction

## 📄 License

MIT License

You may freely use, modify, and distribute this project for personal or commercial purposes.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please give it a star!**
