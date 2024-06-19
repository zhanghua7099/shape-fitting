from __future__ import annotations

import cy_fit_hough as m
import numpy as np

def test_version():
    assert m.__version__ == "0.0.1"


def test_add():
    assert m.add(1, 2) == 3


def test_sub():
    assert m.subtract(1, 2) == -1

def test_inv():
    assert m.inv(np.eye(4))