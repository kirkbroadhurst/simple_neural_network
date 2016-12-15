import numpy as np
from lib.network import build_synapses


def test_one_hidden_layer():
    """
    Tests that a trivial case with one hidden layer with one output will work
    :return:
    """
    input_data = np.matrix(((1, 2, 3), (4, 5, 6)))
    synapses = build_synapses(input_data, [2])

    assert len(synapses) == 1
    assert synapses[0].shape == (4, 2)


