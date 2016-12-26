import numpy as np
import os
from lib.network import theta_prime, g_prime
from data import load_csv


def test_theta_prime():
    a1 = load_csv('a1.csv')
    a2 = load_csv('a2.csv')
    a3 = load_csv('a3.csv')

    z2 = load_csv('z2.csv')
    z3 = load_csv('z3.csv')

    # these are transposed for some reason.
    theta1 = load_csv('theta1.csv').T
    theta2 = load_csv('theta2.csv').T

    theta1_grad = load_csv('theta1_grad.csv').T
    theta2_grad = load_csv('theta2_grad.csv').T

    y = load_csv('y.csv')

    result = theta_prime([a1, a2, a3], [z2, z3], [theta1, theta2], y)
    assert np.allclose(result[0], theta1_grad)
    assert np.allclose(result[1], theta2_grad)


def test_theta_prime_regularization():
    a1 = load_csv('a1.csv')
    a2 = load_csv('a2.csv')
    a3 = load_csv('a3.csv')

    z2 = load_csv('z2.csv')
    z3 = load_csv('z3.csv')

    # these are transposed for some reason.
    theta1 = load_csv('theta1.csv').T
    theta2 = load_csv('theta2.csv').T

    theta1_grad_reg = load_csv('theta1_grad_reg.csv').T
    theta2_grad_reg = load_csv('theta2_grad_reg.csv').T

    y = load_csv('y.csv')

    result = theta_prime([a1, a2, a3], [z2, z3], [theta1, theta2], y, 1)
    assert np.allclose(result[0], theta1_grad_reg)
    assert np.allclose(result[1], theta2_grad_reg)


def test_generic_vs_static():
    a1 = load_csv('a1.csv')
    a2 = load_csv('a2.csv')
    a3 = load_csv('a3.csv')

    z2 = load_csv('z2.csv')
    z3 = load_csv('z3.csv')

    # these are transposed for some reason.
    theta1 = load_csv('theta1.csv').T
    theta2 = load_csv('theta2.csv').T

    y = load_csv('y.csv')

    result = theta_prime([a1, a2, a3], [z2, z3], [theta1, theta2], y)
    result2 = fixed_theta_prime([a1, a2, a3], [z2, z3], [theta1, theta2], y)
    assert np.allclose(result[0], result2[0])
    assert np.allclose(result[1], result2[1])


def fixed_theta_prime(a, z, theta, y):
    """
    Fixed length / non-generic version of the theta_prime method (to compare results / explain logic without loops)
    Compute the gradient for the theta terms with respect to the cost
    :param a: Array of input / post-sigmoid matrices : a_(n+1) = sigmoid(z_n)
    :param z: Array of interim matrices : z_n = a_n * theta_n
    :param theta: Array of coefficient matrices
    :param y: Expected output, i.e. labels
    :return:
    """

    m = y.shape[0]

    # hard coding to the number of layers, for simplicity
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]

    theta1 = theta[0]
    theta2 = theta[1]

    z2 = z[0]
    z3 = z[1]

    d3 = a3 - y
    d2 = np.multiply((d3 * theta2.T[:, 1:]), g_prime(z2))

    delta1 = a1.T*d2
    delta2 = a2.T*d3

    theta1_prime = delta1 / m
    theta2_prime = delta2 / m

    return [theta1_prime, theta2_prime]


if __name__ == "__main__":
    a = os.path.realpath(__file__)
    test_generic_vs_static()
    test_theta_prime()
    test_theta_prime_regularization()
