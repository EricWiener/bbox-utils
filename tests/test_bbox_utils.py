# -*- coding: utf-8 -*-
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
