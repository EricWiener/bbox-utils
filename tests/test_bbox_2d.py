# -*- coding: utf-8 -*-
import pytest

import numpy as np

from bbox_utils import BoundingBox

__author__ = "Eric Wiener"
__copyright__ = "Eric Wiener"
__license__ = "mit"


def test_bounding_box_init():
    """Make sure that the points are sorted correctly"""
    tl = np.array([1.0, 1.0])
    tr = np.array([2.0, 1.0])
    bl = np.array([1.0, 3.0])
    br = np.array([2.0, 3.0])
    unordered_points = np.array([tl, br, bl, tr])
    ordered_points = np.array([tl, tr, br, bl])
    box = BoundingBox(unordered_points)
    assert np.array_equal(box.points, ordered_points)


def test_bounding_to_xywh():
    """Simple case to see if the box can be converted to xywh"""
    tl = np.array([1.0, 1.0])
    tr = np.array([2.0, 1.0])
    bl = np.array([1.0, 3.0])
    br = np.array([2.0, 3.0])
    unordered_points = np.array([tl, br, bl, tr])
    box = BoundingBox(unordered_points)
    xy, w, h = box.to_xywh()
    assert np.array_equal(tl, xy)
    assert w == 1.0
    assert h == 2.0


def test_bounding_to_xyxy():
    """Simple case to see if the box can be converted to xyxy"""
    tl = np.array([1.0, 1.0])
    tr = np.array([2.0, 1.0])
    bl = np.array([1.0, 3.0])
    br = np.array([2.0, 3.0])
    unordered_points = np.array([tl, br, bl, tr])
    box = BoundingBox(unordered_points)
    xy1, xy2 = box.to_xyxy()
    assert np.array_equal(tl, xy1)
    assert np.array_equal(br, xy2)


def test_box_from_xyxy():
    """Test to see if bounding box can be created from XYXY format"""
    xy1 = np.array([2369, 2975])
    xy2 = np.array([2369 + 40, 2975 + 70])
    bbox = BoundingBox.from_xyxy(xy1, xy2)
    xy, w, h = bbox.to_xywh()
    assert np.array_equal(xy, xy1)
    assert w == 70
    assert h == 40


def test_box_from_xyxy_swapped_order():
    """Test to see if bounding box can be created from XYXY
    format when top_left and bottom_right are swapped"""
    xy1 = np.array([2369, 2975])
    xy2 = np.array([2369 + 40, 2975 + 70])
    bbox = BoundingBox.from_xyxy(xy2, xy1)
    xy, w, h = bbox.to_xywh()
    assert np.array_equal(xy, xy1)
    assert w == 70
    assert h == 40


def test_box_from_xyxy_horizontal_line():
    """Test to see assertion thrown with invalid points (horizontal line)"""
    xy1 = np.array([2369, 2975])
    xy2 = np.array([2369, 2975 + 70])

    with pytest.raises(AssertionError):
        _ = BoundingBox.from_xyxy(xy1, xy2)

    with pytest.raises(AssertionError):
        _ = BoundingBox.from_xyxy(xy2, xy1)


def test_box_from_xyxy_vertical_line():
    """Test to see assertion thrown with invalid points (vertical line)"""
    xy1 = np.array([2369, 2975])
    xy2 = np.array([2369 + 70, 2975])

    with pytest.raises(AssertionError):
        _ = BoundingBox.from_xyxy(xy2, xy1)

    with pytest.raises(AssertionError):
        _ = BoundingBox.from_xyxy(xy1, xy2)


def test_box_from_xyxy_same_point():
    """Test to see assertion thrown with invalid points (points equal)"""
    xy1 = np.array([2369, 2975])

    with pytest.raises(AssertionError):
        _ = BoundingBox.from_xyxy(xy1, xy1)


def test_box_from_xywh():
    """Test to see if bounding box can be created from XYWH format"""
    xywh = np.array([2369, 2975, 74, 78])
    xy = xywh[:2]
    w = xywh[2]
    h = xywh[3]
    bbox = BoundingBox.from_xywh(xy, w, h)
    xyxy = bbox.to_xyxy()
    assert np.array_equal(xyxy, np.array([[2369, 2975], [2369 + 78, 2975 + 74]]))


def test_bounding_to_yolo():
    """Test to see if bounding box can be converted to YOLO"""
    image_dim = (3648, 5472)
    xywh = np.array([2369, 2975, 74, 78])
    xy = xywh[:2]
    w = xywh[2]
    h = xywh[3]
    bbox = BoundingBox.from_xywh(xy, w, h)
    correct = np.array(
        [
            0.43969298245614036,
            0.8262061403508771,
            0.013523391812865496,
            0.02138157894736842,
        ]
    )
    assert np.array_equal(bbox.to_yolo(image_dim), correct)


def test_bounding_box_center():
    xywh = np.array([2369, 2975, 74, 78])
    xy = xywh[:2]
    w = xywh[2]
    h = xywh[3]
    bbox = BoundingBox.from_xywh(xy, w, h)
    assert np.array_equal(bbox.center, np.array([2406, 3014]))
