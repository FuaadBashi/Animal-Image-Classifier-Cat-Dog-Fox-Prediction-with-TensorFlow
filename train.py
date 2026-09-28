"""Train the cat/dog/fox classifier by transfer learning from VGG16.

Expects one folder per class:

    animal_images_dl/
      train/{cat,dog,fox}/*.jpg
      valid/{cat,dog,fox}/*.jpg

The VGG16 convolutional base (ImageNet weights) is frozen and a small classification head is
trained on top. Inputs get the same ImageNet preprocessing that main.py applies at inference, and
it is applied once, in the data pipeline, so training and prediction see identical pixels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).parent
IMAGE_SIZE = (224, 224)
CLASSES = ["cat", "dog", "fox"]


def build_model(num_classes: int, weights: str | None = "imagenet"):
    from tensorflow import keras

    base = keras.applications.VGG16(
        include_top=False, weights=weights, input_shape=(*IMAGE_SIZE, 3)
    )
    base.trainable = False
    model = keras.Sequential(
        [
            keras.Input(shape=(*IMAGE_SIZE, 3)),
            # Augmentation layers are active only during training.
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            base,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_split(directory: Path, batch_size: int, shuffle: bool):
    import tensorflow as tf
    from tensorflow.keras.applications.imagenet_utils import preprocess_input

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,  # fixed order, so label indices match main.CLASSES
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )
    return dataset.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)


def train(data_dir: Path, output: Path, epochs: int, batch_size: int, weights: str | None):
    from tensorflow import keras

    train_ds = load_split(data_dir / "train", batch_size, shuffle=True)
    valid_ds = load_split(data_dir / "valid", batch_size, shuffle=False)

    model = build_model(len(CLASSES), weights)
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True
            )
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    best = max(history.history["val_accuracy"])
    print(f"Saved {output} (best validation accuracy {best:.3f})")
    return model, best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=HERE / "animal_images_dl")
    parser.add_argument("--output", type=Path, default=HERE / "models/animal_model.keras")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="random initial weights (for offline smoke tests; accuracy will be poor)",
    )
    args = parser.parse_args(argv)
    for split in ("train", "valid"):
        if not (args.data / split).is_dir():
            parser.error(f"Missing {args.data / split}/ with one sub-folder per class")
    train(
        args.data,
        args.output,
        args.epochs,
        args.batch_size,
        None if args.no_pretrained else "imagenet",
    )


if __name__ == "__main__":
    main()
