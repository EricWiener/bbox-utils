# -*- coding: utf-8 -*-
import pytest

import numpy as np

from bbox_utils.utils import point_within_dimensions


def test_point_within_dimensions_true():
    """Test a point within the dimensions"""
    point = np.array([10, 20])
    image_dimensions = np.array([100, 100])
    assert point_within_dimensions(point, image_dimensions)


def test_point_within_dimensions_border():
    """Make sure a point on the non-zero border is rejected as out of bounds"""
    point = np.array([100, 20])
    image_dimensions = np.array([100, 100])
    assert not point_within_dimensions(point, image_dimensions)


def test_point_with_zero_value_is_good():
    """Make sure a point with zero is okay"""
    point = np.array([0, 20])
    image_dimensions = np.array([100, 100])
    assert point_within_dimensions(point, image_dimensions)


def test_point_within_dimensions_invalid_sizes():
    """An assertion should be thrown if the # of dimensions don't match"""
    point = np.array([20, 20, 20])
    image_dimensions = np.array([100, 100])

    with pytest.raises(AssertionError):
        assert not point_within_dimensions(point, image_dimensions)

    point = np.array([20, 20])
    image_dimensions = np.array([100, 100, 100])

    with pytest.raises(AssertionError):
        assert not point_within_dimensions(point, image_dimensions)
