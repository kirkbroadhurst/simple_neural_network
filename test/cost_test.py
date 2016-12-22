import csv
import numpy as np
from lib.network import cost


def load_csv(file_name):
    """
    Load a csv file into a numpy matrix of floats
    :param file_name: file name
    :return:
    """
    with open(file_name, 'r') as f:
        data_iter = csv.reader(f, delimiter=',')
        data = [data for data in data_iter]
    matrix = np.matrix(data, dtype=float)
    return matrix


def test_cost():
    """
    Test using some known data, borrowed from Andrew Ng's Machine Learning Coursera class
    :return:
    """
    y = load_csv('data/y.csv')
    est = load_csv('data/est.csv')
    j = cost(est, y)

    # cost should be 0.287629
    assert 0.28762 < j < 0.28763


def test_cost_regularized():
    """
    Test using some known data, borrowed from Andrew Ng's Machine Learning Coursera class
    :return:
    """
    y = load_csv('data/y.csv')
    est = load_csv('data/est.csv')
    theta1 = load_csv('data/theta1.csv')
    theta2 = load_csv('data/theta2.csv')
    j = cost(est, y, [theta1, theta2], 1)

    # cost should be 0.383770
    assert 0.383769 < j < 0.383771


if __name__ == "__main__":
    test_cost_regularized()
