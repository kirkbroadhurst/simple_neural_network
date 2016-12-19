from lib.network import g
import numpy as np


def test_sigmoid_zero():
    assert g(0) == 0.5


def test_sigmoid_one():
    assert g(1) - 0.73 < 0.01
    assert g(-1) - 0.26 < 0.01


def test_sigmoid_matrix():
    data = np.matrix(((0, 1), (-1, 100)))
    result = g(data)
    print result
    assert result[0, 0] == 0.5
    assert result[0, 1] - 0.73 < 0.01
    assert result[1, 0] - 0.26 < 0.01
    assert result[1, 1] > 0.999
