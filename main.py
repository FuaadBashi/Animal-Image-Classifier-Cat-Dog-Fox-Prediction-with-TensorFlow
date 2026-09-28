import argparse
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "models/animal_model.keras"
TEST_DIR = Path(__file__).parent / "animal_images_dl/test"
CLASSES = ["cat", "dog", "fox"]


def show_image(image_path):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    plt.imshow(mpimg.imread(image_path))
    plt.axis("off")
    plt.show()
    plt.close()


def make_predictions(model, image_path, *, show=False):
    from tensorflow.keras.applications.imagenet_utils import preprocess_input
    from tensorflow.keras.preprocessing import image as image_utils

    if show:
        show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(224, 224))
    batch = np.expand_dims(image_utils.img_to_array(image), axis=0)
    return model.predict(preprocess_input(batch), verbose=0)


def predicted_label(predictions, classes):
    values = np.asarray(predictions)
    if values.shape != (1, len(classes)) or not np.isfinite(values).all():
        raise ValueError("Expected one finite prediction per configured class")
    return classes[int(values[0].argmax())]


def run_tests(model, test_dir, *, classes=CLASSES, show=False):
    images = sorted(
        p
        for p in Path(test_dir).iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise ValueError("No supported images found in the input directory")
    correct = 0
    for path in images:
        label = predicted_label(make_predictions(model, path, show=show), classes)
        print(f"Prediction for {path.name}: {label}")
        correct += label.lower() in path.stem.lower()
    ratio = correct / len(images)
    print(f"Filename-match ratio: {correct}/{len(images)} ({ratio:.4f})")
    return ratio


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a supplied animal classification model")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--images", type=Path, default=TEST_DIR)
    parser.add_argument("--classes", nargs="+", default=CLASSES)
    parser.add_argument("--show", action="store_true", help="Preview each image interactively")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")
    if not args.images.is_dir():
        parser.error(f"Image directory not found: {args.images}")
    if len(set(args.classes)) != len(args.classes):
        parser.error("Class labels must be unique")
    from tensorflow.keras.models import load_model

    try:
        run_tests(load_model(args.model), args.images, classes=args.classes, show=args.show)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
