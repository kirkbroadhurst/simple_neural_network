import numpy as np


def build_synapses(layers):
    """
    Construct synapses to forward propagate between layers.
    These are matrices that will map from layer[n] to layer[n+1]
    :param layers: an array describing the size of each layer (number of nodes).
    The length of the array indicates the number of hidden layers
    Must include the size of the input layer and the output layer, i.e. minimum length == 2
    :return: An array of numpy matrices that describe the synapses of a neural network,
    initialized to mean centered random values x: x<1 & x>-1
    """

    if len(layers) < 2:
        raise Warning('need at least two layers')

    synapses = []
    for l in range(1, len(layers)):
        # for the first layer, the input size is the number of features in training data
        n = layers[l - 1]
        m = layers[l]

        # add a random mean-centered synapse of size; 2 * random(0,1) - 1
        # the synapse is of height n+1 because n inputs plus 1 for bias/constant
        synapses += [2 * np.random.random((n + 1, m)) - 1]
    return synapses
