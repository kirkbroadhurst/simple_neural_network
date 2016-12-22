import numpy as np
from lib.network import theta_prime
from data import load_csv


def test_theta_prime():
    a1 = load_csv('data/a1.csv')
    a2 = load_csv('data/a2.csv')
    a3 = load_csv('data/a3.csv')

    z2 = load_csv('data/z2.csv')
    z3 = load_csv('data/z3.csv')

    theta1 = load_csv('data/theta1.csv')
    theta2 = load_csv('data/theta2.csv')

    theta1_grad = load_csv('data/theta1_grad.csv')
    theta2_grad = load_csv('data/theta2_grad.csv')

    y = load_csv('data/y.csv')

    result = theta_prime([a1, a2, a3], [z2, z3], [theta1, theta2], y)
    assert np.allclose(result[0], theta1_grad)
    assert np.allclose(result[1], theta2_grad)


if __name__ == "__main__":
    test_theta_prime()
