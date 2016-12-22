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

    data = [initial_layer]

    # for each synapse, push through in_data and get out_data
    for i in range(len(thetas)):
        x = data[i]
        theta = thetas[i]
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
    return g(x)*(1-g(x))


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


def cost(estimated, y, thetas=[], l=0):
    """
    Compute the cost of a neural network vs the expect / label output
    :param estimated: The output of the final layer of a neural network - the estimation
    :param y: The expected value for output of the neural network - the label
    :param thetas: Coefficients used in neural network (regularization)
    :param l: lambda value - regularization parameter
    :return: Cost of the network
    """

    # the number of observations
    m = y.shape[0]

    # take log of the estimated value so that the 'cost' of predicting 1 is 0, and the 'cost' of predicting zero -> inf
    # multiply by y, i.e. when y = 1 we should have estimated 1; an estimate closer to zero is high cost i.e. log(0)
    # -y * log(estimated)

    # the reverse applies for y = 0; we want 'high cost' when y = 0 and estimation -> 1; so use log(1 - est) -> inf.
    # and multiply by 1 - y, i.e. 1 when y == 0

    gap = np.multiply(-y, np.log(estimated)) - np.multiply(1-y, np.log(1-estimated))
    j = 1.0 / m * np.sum(gap)

    # if thetas are supplied for regularization, return sum of squares
    for t in thetas:
        # remove the bias / constant term
        t_ = t[:, 1:]
        j += l/(2.0*m) * np.sum(np.multiply(t_, t_))

    return j




