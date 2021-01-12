# -*- coding: utf-8 -*-
import numpy as np

from bbox_utils.utils import order_points

__author__ = "Eric Wiener"
__copyright__ = "Eric Wiener"
__license__ = "mit"


def test_order_points():
    tl = np.array([1.0, 1.0])
    tr = np.array([2.0, 1.0])
    bl = np.array([1.0, 3.0])
    br = np.array([2.0, 3.0])
    unordered_points = np.array([tl, br, bl, tr])
    ordered_points = np.array([tl, tr, br, bl])
    output = order_points(unordered_points)
    assert np.array_equal(output, ordered_points)
