import numpy as np
from lib.network import build_synapses, forward_propagate, cost, theta_prime, softmax


class SimpleMachine(object):
    
    def __init__(self, training_data, result, layers, theta=[], reg=0.01):
        """
        Create a simple neural network machine for classification
        :param training_data: input data as numpy matrix, n rows of observations with m features
        :param result: observed results, n rows of a single value
        :param layers: an array where each value indicates the size of a hidden layer
        :param theta: existing (starting) set of coefficients to use. Default = []
        :param reg: regularization parameter, aka lambda, to apply. Default = 0.01
        """
        self.training_data = training_data        
        self.Y = result
        self.layers = layers
        if theta == []:
            self.theta = build_synapses(layers)
        else:
            self.theta = theta
        self.l = reg

    @property
    def size(self):
        return len(self.theta)

    @property
    def m(self):
        """
        Number of rows in the training data
        :return:
        """
        return self.training_data.shape[0]

    def score(self, data):
        """
        Score / predict the input data using the machine
        :param data: Data set to predict (obviously same shape as training data)
        :return: Predictions in the same shape as 'result' / labels
        """
        _, z_0 = forward_propagate(data, self.theta)
        return softmax(z_0[-1])

    def train(self, iterations=10000, quiet=False):
        """
        Train a machine
        :param iterations:
        :return:
        """

        for ii in range(iterations):
            i = ii % self.m
            test = i % self.m
            x = self.training_data[test]
            y = self.Y[test]

            (a_, z_) = forward_propagate(x, self.theta)

            s = softmax(z_[-1])

            # some hacks to monitor progress
            # uncomment this to see progress
            '''
            if i % 1000 == 0 and not quiet:
                (a_0, z_0) = forward_propagate(self.training_data, self.theta)
                print 'cost', cost(a_0[-1], self.Y)
                values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                m = s.max()
                est = s.tolist()[0].index(m)
                actual = int((values * self.Y[i].T)[0, 0])
                print est, '(actual = {0})'.format(actual), "- {0:.0f}% confidence".format(100 * s[0, est])
            '''

            gradients = theta_prime(a_, z_, self.theta, y)

            for (ix, g) in enumerate(gradients):
                self.theta[ix] -= self.l * g

        (a_final, z_final) = forward_propagate(self.training_data, self.theta)
        print 'cost', cost(a_final[-1], self.Y)
        return self.theta

