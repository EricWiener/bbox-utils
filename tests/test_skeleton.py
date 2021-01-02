# -*- coding: utf-8 -*-

import pytest

from bbox_utils.skeleton import fib

__author__ = "Eric Wiener"
__copyright__ = "Eric Wiener"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
