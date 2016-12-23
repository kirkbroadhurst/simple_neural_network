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


def forward_propagate(a1, thetas):
    """
    Push input data through the neural network
    :param a1: Initial layer of data (raw features)
    :param thetas: Set of thetas to act in synapses
    :return: Tuple of a and z matrices;
         a_(n+1) = sigmoid(z_n)
         z_n = a_n * theta_n
    """

    a_ = [a(a1)]
    z_ = []

    # for each synapse, push through in_data and get out_data
    for i in range(len(thetas)):
        x = a_[i]
        theta = thetas[i]
        z_.append(z(x, theta))
        output = g(z_[-1])
        a_.append(a(output))
    # the last a value shouldn't have bias term - it's the network output. Truncate the bias column
    a_[-1] = a_[-1][:, 1:]
    return a_, z_


def a(x):
    """
    Compute the a value for x with additional bias input
    :param x: the input value for a matrix computation
    :return: X with appended column of 1s for bias
    """
    # add extra column for bias/constant term
    a_ = np.matrix(np.empty((x.shape[0], x.shape[1] + 1)))
    a_[:, 0] = 1
    a_[:, 1:] = x
    return a_


def z(a_, theta):
    """
    Compute the z value of a*theta; does not append the bias term
    :param a_: The input data to this synapse
    :param theta: The theta value for this synapse
    :return: a*theta
    """
    return a_.dot(theta)


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
    return np.multiply(g(x), (1-g(x)))


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


def theta_prime(a, z, theta, y):
    """
    Compute the gradient for the theta terms with respect to the cost
    :param a: Array of input / post-sigmoid matrices : a_(n+1) = sigmoid(z_n)
    :param z: Array of interim matrices : z_n = a_n * theta_n
    :param theta: Array of coefficient matrices
    :param y: Expected output, i.e. labels
    :return:
    """

    m = y.shape[0]

    # the last a value is the network output
    out = a[-1]

    # compute d values
    # the first computed d value is the network output - y
    d = [out - y]
    i = len(theta) - 1
    while len(d) < len(theta):
        # d value that is currently first is the last layer we 'backpropagated'
        d_ = d[0]
        t_ = theta[i]
        i -= 1

        # theta[i] corresponds to z[i-1] - there is no theta[L] and there is no z[0]
        z_ = z[i]

        # compute the d value for this layer
        new_value = np.multiply((d_ * t_[:, 1:]), g_prime(z_))
        d.insert(0, new_value)

    big_delta = []
    theta_p = []
    # compute big delta value & theta_prime values
    for (ix, d_) in enumerate(d):
        # the 'first' d value is d2; multiply it by the first a value a1 - and so on
        big_delta.append(d_.T * a[ix])
        theta_p.append(big_delta[-1] / m)

    return theta_p

