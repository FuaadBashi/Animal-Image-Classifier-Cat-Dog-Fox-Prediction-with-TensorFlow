# Animal Image Classifier — Cat, Dog or Fox

[![CI](https://github.com/FuaadBashi/Animal-Image-Classifier-Cat-Dog-Fox-Prediction-with-TensorFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/FuaadBashi/Animal-Image-Classifier-Cat-Dog-Fox-Prediction-with-TensorFlow/actions/workflows/ci.yml)

An image classifier that tells cats, dogs and foxes apart, built by transfer learning from VGG16
with TensorFlow/Keras. `train.py` trains and saves the model; `main.py` classifies a folder of
images with it.

```
$ python main.py --images photos/
Prediction for cat_0.jpg: cat
Prediction for dog_1.jpg: dog
Prediction for fox_0.jpg: fox
Filename-match ratio: 3/3 (1.0000)
```

## How it works

- **Transfer learning.** The VGG16 convolutional base, pre-trained on ImageNet, is frozen. It
  already recognises edges, textures and animal parts. Only a small head is trained on top: global
  average pooling, dropout, and a 3-way softmax. This learns quickly from a few hundred images.
- **Augmentation.** Random horizontal flips and small rotations during training reduce
  overfitting.
- **One preprocessing path.** Images are resized to 224×224 and given VGG16's ImageNet
  preprocessing exactly once, in the training data pipeline and again at inference. Getting this
  wrong, by skipping it or applying it twice, is a classic silent accuracy killer.
- **Fixed class order.** Labels are always `cat, dog, fox`, matching the output units, so
  predictions can't be silently shuffled by folder order.
- **Early stopping.** Training stops when validation accuracy stops improving and keeps the best
  weights.

## Getting started

Requires Python 3.10+.

```bash
git clone https://github.com/FuaadBashi/Animal-Image-Classifier-Cat-Dog-Fox-Prediction-with-TensorFlow.git
cd Animal-Image-Classifier-Cat-Dog-Fox-Prediction-with-TensorFlow
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Arrange your images one folder per class, then train and predict:

```
animal_images_dl/
├── train/{cat,dog,fox}/*.jpg
├── valid/{cat,dog,fox}/*.jpg
└── test/*.jpg                 # e.g. cat_01.jpg, fox_07.jpg
```

```bash
python train.py                  # saves models/animal_model.keras
python main.py                   # classifies animal_images_dl/test
python main.py --images my_photos/ --show
```

`main.py --model` also accepts an existing `.h5` model. When file names contain the true label,
`main.py` reports how many predictions matched.

## Tests

```bash
pip install pytest ruff numpy
pytest                      # fast tests; the training test skips without TensorFlow
pip install -r requirements.txt && pytest   # adds an end-to-end train → save → predict test
ruff format --check . && ruff check .
```
