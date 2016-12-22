import numpy as np
from lib.network import build_synapses, forward_propagate, delta, cost, a


class SimpleMachine(object):
    
    def __init__(self, training_data, result, layers):
        """
        Create a simple neural network machine for classification
        :param training_data: input data as numpy matrix, n rows of observations with m features
        :param result: observed results, n rows of a single value
        :param layers: an array where each value indicates the size of a hidden layer
        """
        self.training_data = training_data        
        self.Y = result
        self.layers = layers
        self.theta = build_synapses(layers)
        self.l = 0.01

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
        Score the input data against the synapses / machine
        :param data:
        :return:
        """
        return forward_propagate(data, self.theta)[-1]

    def train(self, iterations=10000):
        """
        Train a machine
        :param iterations:
        :return:
        """

        np.random.seed(1)

        for i in range(iterations):
            test = i % self.m
            x = self.training_data[test]
            y = self.Y[test]

            (g, z_) = forward_propagate(x, self.theta)
            # backward propagate deltas

            # we need an error matrix for each layer
            errors = [None] * len(g)

            # set the last error value to the actual values minus the last layer
            errors[-1] = y - g[-1]

            # reverse iterate through layers, adjusting synapse according to error
            for th in range(len(self.theta) - 1, -1, -1):
                error = delta(self.theta[th], errors[th+1], g[th])
                self.theta[th] -= error
                errors[th] = error

            big_delta = sum([(a(g[e])*errors[e])[0, 0] for e in range(len(errors) - 1)])
            print big_delta

            d0 = 1.0 / self.m * big_delta
            d = [np.matrix(1.0 / self.m * (big_delta + (self.l * th))) for th in self.theta]
            self.theta = [th + d[ix] for ix, th in enumerate(self.theta)]
            print 'big delta', big_delta
            print 'cost', cost(self.score(self.training_data), self.Y)


if __name__ == "__main__":
    t = np.matrix(((1.0, 0, 0, 0.99), (0, 0.8, 0, 0.95), (0, 0, 0.9, 0.9)))
    l = [4, 3]
    r = np.matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    s = SimpleMachine(t, r, l)
    s.train(100)
