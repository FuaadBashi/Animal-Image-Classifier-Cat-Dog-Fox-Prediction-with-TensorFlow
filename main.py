# test_animals.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model


# Path to your saved Keras/TensorFlow model
MODEL_PATH = "models/animal_model.h5"      

# Path to your test directory with images
TEST_DIR = "animal_images_dl/test"     

# Class labels in the same order as your model's output
CLASSES = ["cat", "dog", "fox"]           



def show_image(image_path: str) -> None:
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def make_predictions(model, image_path: str) -> np.ndarray:
    """
    Load an image, preprocess it, and return the model predictions.
    Assumes input size (224, 224, 3).
    """
    # Optional: show the image being predicted
    show_image(image_path)

    # Load and preprocess
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1, 224, 224, 3)
    image = preprocess_input(image)

    # Predict
    preds = model.predict(image)
    return preds


def run_tests(model, test_dir: str) -> None:
    """
    Loop through all images in test_dir, make predictions,
    and calculate a simple filename-based accuracy.
    """
    total_correct = 0
    total_images = 0

    for filename in os.listdir(test_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_path = os.path.join(test_dir, filename)

            # Get model predictions
            predictions = make_predictions(model, image_path)
            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASSES[predicted_index]

            print(f"Prediction for {filename}: {predicted_class}")
            total_images += 1

            # Very simple check: does the filename contain the class name?
            # e.g. "cat_001.jpg" should contain "cat"
            if predicted_class.lower() in filename.lower():
                total_correct += 1

    if total_images == 0:
        print("No images found in test directory.")
        return

    accuracy = total_correct / total_images
    print(f"\nTotal images: {total_images}")
    print(f"Correct by filename: {total_correct}")
    print(f"Accuracy (by filename match): {accuracy:.4f}")



if __name__ == "__main__":
    # Load the model
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # Run tests
    print(f"Running tests on directory: {TEST_DIR}")
    run_tests(model, TEST_DIR)
