from lib.network import build_synapses


def test_zero_hidden_layer():
    """
    Tests that a trivial case with one hidden layer with one output will work
    :return:
    """
    synapses = build_synapses([3, 2])

    assert len(synapses) == 1
    assert synapses[0].shape == (4, 2)


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

