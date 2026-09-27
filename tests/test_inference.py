import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('animal_inference', Path(__file__).parents[1] / 'main.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


@pytest.mark.parametrize('prediction', [[[1, 2]], [[1, np.nan, 0]], [1, 2, 3], [[1, 2, 3], [3, 2, 1]]])
def test_invalid_model_outputs_are_rejected(prediction):
    with pytest.raises(ValueError):
        app.predicted_label(prediction, app.CLASSES)


def test_class_order_is_respected():
    assert app.predicted_label([[0.1, 0.8, 0.1]], ['fox', 'cat', 'dog']) == 'cat'


def test_empty_directory_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='No supported images'):
        app.run_tests(None, tmp_path)


def test_images_are_sorted_and_subdirectories_ignored(tmp_path, monkeypatch):
    (tmp_path / 'dog.jpg').touch()
    (tmp_path / 'cat.png').touch()
    (tmp_path / 'fox.jpg').mkdir()
    seen = []
    def predict(model, path, *, show):
        seen.append(path.name)
        assert not show
        return [[1, 0, 0]] if path.stem == 'cat' else [[0, 1, 0]]
    monkeypatch.setattr(app, 'make_predictions', predict)
    assert app.run_tests(None, tmp_path) == 1
    assert seen == ['cat.png', 'dog.jpg']


def test_missing_model_fails_before_tensorflow_import(tmp_path):
    with pytest.raises(SystemExit) as error:
        app.main(['--model', str(tmp_path / 'missing.h5')])
    assert error.value.code == 2
