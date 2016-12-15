import numpy as np


def build_synapses(input_layer, layers):
    """
    Construct synapses to forward propagate between layers.
    These are matrices that will map from layer[n] to layer[n+1]
    :param input_layer: the initial input layer, defines the input to the first synapse
    :param layers: an array describing the size of each layer (number of nodes).
    The length of the array indicates the number of hidden layers
    :return: An array of numpy matrices that describe the synapses of a neural network,
    initialized to mean centered random values x: x<1 & x>-1
    """
    synapses = []
    for l in range(len(layers)):
        # for the first layer, the input size is the number of features in training data
        n = input_layer.shape[1] if l == 0 else layers[l - 1]
        m = layers[l]

        # add a random mean-centered synapse of size; 2 * random(0,1) - 1
        # the synapse is of height n+1 because n inputs plus 1 for bias/constant
        synapses += [2 * np.random.random((n + 1, m)) - 1]
    return synapses
