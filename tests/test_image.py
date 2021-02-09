import numpy as np
import pytest

from bbox_utils.image import Image


def test_image_load_from_file():
    # Make sure no errors are thrown
    _ = Image.load_from_file("./tests/assets/buoys.png")


def test_image_constructor():
    # Make sure no errors are thrown
    img = Image.load_from_file("./tests/assets/buoys.png")
    _ = Image(img.image)


def test_image_constructor_invalid():
    with pytest.raises(TypeError):
        _ = Image("no bueno")

    with pytest.raises(TypeError):
        _ = Image(5)

    with pytest.raises(ValueError):
        _ = Image(np.array([]))
