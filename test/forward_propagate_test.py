import numpy as np
from lib.network import forward_propagate


def test_single_synapse():
    """
    Tests that an input matrix with a single synapse can be propagated to an output layer
    :return:
    """
    input_data = np.ones((4, 6))
    synapses = [np.ones((7, 4))]
    (a, z) = forward_propagate(input_data, synapses)

    assert len(a) == 2
    assert (a[0] == input_data).all()
    shape = a[1].shape

    assert shape == (4, 4)

    print a[1]
    # values are all between 0.997527 and 0.997528
    assert (np.zeros(shape) < (a[1] - 0.9990889)).all()
    assert (np.zeros(shape) > (a[1] - 0.9990890)).all()

    assert len(z) == 1
    assert z[0].shape == a[1].shape


def test_three_synapse():
    """
    Tests that an input matrix with a single synapse can be propagated to an output layer
    :return:
    """
    input_data = np.ones((4, 100))
    synapses = [np.ones((101, 50)), np.ones((51, 20)), np.ones((21, 5))]
    (a, z) = forward_propagate(input_data, synapses)

    assert len(a) == 4
    assert (a[0] == input_data).all()

    shape = a[1].shape
    assert shape == (4, 50)
    assert (np.zeros(shape) < (a[1] - 0.99999999)).all()
    assert (np.zeros(shape) > (a[1] - 1.00000001)).all()

    print a[2]
    shape = a[2].shape
    assert shape == (4, 20)
    assert (np.zeros(shape) < (a[2] - 0.99999999)).all()
    assert (np.zeros(shape) > (a[2] - 1.00000001)).all()

    shape = a[3].shape
    assert shape == (4, 5)
    assert (np.zeros(shape) < (a[3] - 0.99999999)).all()
    assert (np.zeros(shape) > (a[3] - 1.00000001)).all()

    assert len(z) == 3
    assert z[0].shape == a[1].shape
    assert z[1].shape == a[2].shape
    assert z[2].shape == a[3].shape
