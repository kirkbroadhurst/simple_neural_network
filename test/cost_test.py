from lib.network import cost
from data import load_csv


def test_cost():
    """
    Test using some known data, borrowed from Andrew Ng's Machine Learning Coursera class
    :return:
    """
    y = load_csv('data/y.csv')
    est = load_csv('data/a3.csv')
    j = cost(est, y)

    # cost should be 0.287629
    assert 0.28762 < j < 0.28763


def test_cost_regularized():
    """
    Test using some known data, borrowed from Andrew Ng's Machine Learning Coursera class
    :return:
    """
    y = load_csv('data/y.csv')
    a3 = load_csv('data/a3.csv')
    theta1 = load_csv('data/theta1.csv')
    theta2 = load_csv('data/theta2.csv')
    j = cost(a3, y, [theta1, theta2], 1)

    # cost should be 0.383770
    assert 0.383769 < j < 0.383771


if __name__ == "__main__":
    test_cost_regularized()
