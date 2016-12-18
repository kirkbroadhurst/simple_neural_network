from lib.network import sigmoid, sigmoid_slope
import numpy as np


def test_sigmoid_zero():
    assert sigmoid(0) == 0.5


def test_sigmoid_one():
    assert sigmoid(1) - 0.73 < 0.01
    assert sigmoid(-1) - 0.26 < 0.01


def test_sigmoid_matrix():
    data = np.matrix(((0, 1), (-1, 100)))
    result = sigmoid(data)
    print result
    assert result[0, 0] == 0.5
    assert result[0, 1] - 0.73 < 0.01
    assert result[1, 0] - 0.26 < 0.01
    assert result[1, 1] > 0.999


def test_sigmoid_second_derivative_zero():
    assert sigmoid_slope(0) == 1.0


def test_sigmoid_second_derivative_one():
    assert sigmoid_slope(1) - 0.73 < 0.01
    assert sigmoid_slope(-1) - 0.26 < 0.01


def test_sigmoid_second_derivative_matrix():
    data = np.matrix(((0, 1), (-1, 100)))
    result = sigmoid_slope(data)
    print result
    assert result[0, 0] == 1.0
    assert result[0, 1] - 0.73 < 0.01
    assert result[1, 0] - 0.26 < 0.01
    assert result[1, 1] < 0.001
