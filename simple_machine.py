import numpy as np
from lib.network import build_synapses, forward_propagate, delta, cost


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

    @property
    def size(self):
        return len(self.theta)

    def score(self, data):
        """
        Score the input data against the synapses / machine
        :param data:
        :return:
        """
        return forward_propagate(data, self.theta)

    def train(self, iterations=10000):
        """
        Train a machine
        :param iterations:
        :return:
        """

        np.random.seed(1)

        for i in range(iterations):
            layers = forward_propagate(self.training_data, self.theta)

            ######
            # backward propagate deltas
            ######
            
            # we need an error matrix for each layer
            errors = [None] * len(layers)

            # set the last error value to the actual values minus the last layer
            errors[-1] = self.Y - layers[-1]

            # reverse iterate through layers, adjusting synapse according to error
            for t in range(len(self.theta) - 1, -1, -1):
                error = delta(self.theta[t], errors[t+1], layers[t])
                self.theta[t] -= error
                errors[t] = error

            print 'cost', cost(layers[-1], self.Y)

        print layers[-1]


if __name__ == "__main__":
    t = np.matrix(((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)))
    l = [4, 2]
    r = np.matrix(((1, 0), (0, 1), (1, 0)))
    s = SimpleMachine(t, r, l)
    s.train(5)
