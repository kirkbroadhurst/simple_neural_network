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


def forward_propagate(initial_layer, thetas):
    """
    Push input data through the neural network
    :param initial_layer: Initial layer of data (raw features)
    :param thetas: Set of thetas to act in synapses
    :return: Array of input matrices and output matrices
    """

    #TODO remove this a stuff
    data = [initial_layer]
    #a_ = []

    # for each synapse, push through in_data and get out_data
    for i in range(len(thetas)):
        x = data[i]
        theta = thetas[i]
        #a_.append(a(x))
        output = g(z(x, theta))
        data.append(np.matrix(output))
    return data


def a(x):
    """
    Compute the a value for x with additional bias input
    :param x: the input value for a matrix computation
    :return: X with appended column of 1s for bias
    """
    # add extra column for bias/constant term
    a_ = np.matrix(np.empty((x.shape[0], x.shape[1] + 1)))
    a_[:, -1] = 1
    a_[:, :-1] = x
    return a_


def z(x, theta):
    """
    Compute the z value of x*theta; append the bias term
    :param x: The input data to this synapse
    :param theta: The theta value for this synapse
    :return: x*theta including the bias term
    """
    return a(x).dot(theta)


def g(x):
    """
    Get the value of the sigmoid function at x
    :param x: Scalar, matrix, array etc of real values
    :return: The value of the sigmoid function at point(s) x
    """
    return 1/(1+np.exp(-x))


def g_prime(x):
    """
    Get the derivative of the sigmoid function at x
    :param x: Scalar, matrix, array etc of real values
    :return: The derivative of the sigmoid function at point(s) x
    """
    return x*(1-x)


def error(output, expected):
    """
    Get the error at a particular layer
    :param output: The output of the layer
    :param expected: The expect value for that layer
    :return:
    """
    return output - expected


def delta(theta, output_error, activation):
    """
    Get the error for a hidden layer (no explict labels)
    :param theta: The synapse to which this layer is an input
    :param output_error: The upstream error, backed out with the synapse
    :param activation: The activation for this layer
    :return: delta for the hidden layer
    """

    # construct the activation matrix with an addition bias term
    a_ = a(activation)
    slope = np.multiply(a_, 1 - a_)
    result = np.multiply(theta.dot(output_error.T), slope.T)
    return result


def cost(output, expected):
    """
    Compute the cost of a neural network vs the expect / label output
    :param output: The output of the final layer of a neural network
    :param expected: The expected value for that layer
    :return: Cost of the network
    """
    gap = error(output, expected)
    n = expected.shape[0]
    p = np.multiply(gap, gap)
    c = 1.0/(2*n)*sum(sum(p).T)
    return c[0, 0]




