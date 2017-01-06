import numpy as np
from simplenet.lib.network import forward_propagate


def test_single_synapse():
    """
    Tests that an input matrix with a single synapse can be propagated to an output layer
    :return:
    """
    input_data = np.ones((4, 6))
    synapses = [np.ones((7, 4))]
    (a, z) = forward_propagate(input_data, synapses)

    assert len(a) == 2
    assert (a[0][:, 1:] == input_data).all()
    assert a[0][:, 1:].shape == input_data.shape

    shape = a[1].shape
    assert shape == (4, 4)

    # values are all between 0.997527 and 0.997528
    assert (np.zeros(shape) < (a[1] - 0.9990889))[:, 1:].all()
    assert (np.zeros(shape) > (a[1] - 0.9990890))[:, 1:].all()

    assert len(z) == 1
    assert z[0].shape[0] == a[1].shape[0]
    assert z[0].shape[1] == a[1].shape[1]


def test_three_synapse():
    """
    Tests that an input matrix with a single synapse can be propagated to an output layer
    :return:
    """
    input_data = np.ones((4, 100))
    synapses = [np.ones((101, 50)), np.ones((51, 20)), np.ones((21, 5))]
    (a, z) = forward_propagate(input_data, synapses)

    assert len(a) == 4
    assert (a[0][:, 1:] == input_data).all()

    shape = a[1].shape
    assert shape == (4, 51)
    assert (np.zeros(shape) < (a[1] - 0.99999999))[:, 1:].all()
    assert (np.zeros(shape) > (a[1] - 1.00000001))[:, 1:].all()

    print a[2]
    shape = a[2].shape
    assert shape == (4, 21)
    assert (np.zeros(shape) < (a[2] - 0.99999999))[:, 1:].all()
    assert (np.zeros(shape) > (a[2] - 1.00000001))[:, 1:].all()

    shape = a[3].shape
    assert shape == (4, 5)
    assert (np.zeros(shape) < (a[3] - 0.99999999))[:, 1:].all()
    assert (np.zeros(shape) > (a[3] - 1.00000001))[:, 1:].all()

    # a matrices have an extra column appended for the bias term
    assert len(z) == 3
    assert z[0].shape[0] == a[1].shape[0]
    assert z[0].shape[1] + 1 == a[1].shape[1]

    assert z[1].shape[0] == a[2].shape[0]
    assert z[1].shape[1] + 1 == a[2].shape[1]

    # except the last non-input a matrix
    assert z[2].shape[0] == a[3].shape[0]
    assert z[2].shape[1] == a[3].shape[1]


if __name__ == "__main__":
    test_single_synapse()
    test_three_synapse()
