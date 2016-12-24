from lib.network import build_synapses


def assert_layer_result(layers, shape, length):
    """
    Boilerplate function to leverage common testing / different shapes
    :param layers: Layers parameter passed to method
    :param shape: Expected resulting shape(s)
    :param length: Total number of values to inspect in all synapses
    :return:
    """
    synapses = build_synapses(layers)
    assert len(synapses) == len(layers) - 1

    for s in range(len(synapses)):
        assert synapses[s].shape == shape[s]

    avg = 0
    for s in synapses:
        for row in s:
            for col in row:
                assert 1 > col > -1
                avg += col/length

    assert 0.5 > avg > -0.5


def test_two_layers():
    layers = [3, 2]
    shape = [(4, 2)]
    length = 8
    assert_layer_result(layers, shape, length)


def test_zero_hidden_layer():
    """
    Tests that a trivial case with one hidden layer with one output will work
    :return:
    """
    synapses = build_synapses([3, 2])

    assert len(synapses) == 1
    assert synapses[0].shape == (4, 2)

    avg = 0
    for row in synapses[0]:
        for col in row:
            assert 1 > col > -1
            avg += col/8

    assert 1 > avg > -1


def test_no_layers():
    """
    Tests not providing any layers will throw error
    :return:
    """
    try:
        build_synapses([])
    except Warning as e:
        print e
        return

    return False


def test_one_layer():
    """
    Tests providing fewer than two layers will throw error
    :return:
    """
    try:
        build_synapses([5])
    except Warning as e:
        print e
        return

    return False


if __name__ == "__main__":
    test_zero_hidden_layer()
