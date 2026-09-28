"""End-to-end: train a tiny model, save it, and classify with main.py.

Skipped unless TensorFlow is installed, so the default test run stays fast.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
Image = pytest.importorskip("PIL.Image")

ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_images(directory: Path, cls: str, count: int, rng):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(directory / f"{cls}_{i}.jpg")


def test_a_trained_model_is_saved_and_predicts_one_known_class_per_image(tmp_path):
    train, app = load("train"), load("main")
    rng = np.random.default_rng(0)
    for cls in train.CLASSES:
        write_images(tmp_path / "data/train" / cls, cls, 2, rng)
        write_images(tmp_path / "data/valid" / cls, cls, 1, rng)
    write_images(tmp_path / "test", "cat", 2, rng)
    model_path = tmp_path / "model.keras"

    train.main(
        ["--data", str(tmp_path / "data"), "--output", str(model_path), "--epochs", "1",
         "--batch-size", "4", "--no-pretrained"]
    )  # fmt: skip

    assert model_path.is_file()
    model = tf.keras.models.load_model(model_path)
    assert model.output_shape == (None, len(app.CLASSES))
    ratio = app.run_tests(model, tmp_path / "test")
    assert 0.0 <= ratio <= 1.0
