import numpy as np
from lib.network import forward_propagate


def test_single_synapse():
    """
    Tests that an input matrix with a single synapse can be propagated to an output layer
    :return:
    """
    input_data = np.ones((4, 6))
    synapses = [np.ones((7, 4))]
    output = forward_propagate(input_data, synapses)

    assert len(output) == 2
    assert (output[0] == input_data).all()
    shape = output[1].shape

    assert shape == (4, 4)

    # values are all between 0.997527 and 0.997528
    assert (np.zeros(shape) < (output[1] - 0.997527)).all()
    assert (np.zeros(shape) > (output[1] - 0.997528)).all()


def test_three_synapse():
    """
    Tests that an input matrix with a single synapse can be propagated to an output layer
    :return:
    """
    input_data = np.ones((4, 100))
    synapses = [np.ones((101, 50)), np.ones((51, 20)), np.ones((21, 5))]
    output = forward_propagate(input_data, synapses)

    assert len(output) == 4
    assert (output[0] == input_data).all()

    shape = output[1].shape
    assert shape == (4, 50)
    assert (np.zeros(shape) < (output[1] - 0.99999999)).all()
    assert (np.zeros(shape) > (output[1] - 1.00000001)).all()

    print output[2]
    shape = output[2].shape
    assert shape == (4, 20)
    assert (np.zeros(shape) < (output[2] - 0.99999999)).all()
    assert (np.zeros(shape) > (output[2] - 1.00000001)).all()

    shape = output[3].shape
    assert shape == (4, 5)
    assert (np.zeros(shape) < (output[3] - 0.99999999)).all()
    assert (np.zeros(shape) > (output[3] - 1.00000001)).all()
