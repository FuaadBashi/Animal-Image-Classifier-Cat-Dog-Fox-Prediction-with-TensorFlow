# Animal Image Classification — TensorFlow Inference

A Python inference script for a separately trained cat/dog/fox classifier. It loads images, resizes them to 224 × 224, applies preprocessing, and prints predicted labels.

## Setup

The repository contains [main.py](main.py). **Model weights and test images are not included.** Supply a compatible Keras model and ensure its preprocessing and output-class order match the script.

```bash
git clone https://github.com/FuaadBashi/Animal-Image-Classifier-Cat-Dog-Fox-Prediction-with-TensorFlow.git
cd Animal-Image-Classifier-Cat-Dog-Fox-Prediction-with-TensorFlow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install tensorflow matplotlib numpy pillow
```

Set `MODEL_PATH`, `TEST_DIR`, and `CLASSES` in `main.py`, then run:

```bash
python main.py
```

The defaults expect `models/animal_model.h5` and `animal_images_dl/test`, with classes ordered as `cat`, `dog`, `fox`.

## Code to explore

- `show_image`: previews each image with Matplotlib; close the preview to continue.
- `make_predictions`: prepares one image and calls the model.
- `run_tests`: iterates over supported image files and prints predictions.

## Evaluation boundary

The reported filename-match ratio checks whether the predicted class appears in the filename. It is a convenience check, not a held-out accuracy benchmark. No measured model-performance claim is made here; a labeled evaluation set and training provenance are needed for that.
