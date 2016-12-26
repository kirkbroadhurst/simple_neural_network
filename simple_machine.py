import numpy as np
from lib.network import build_synapses, forward_propagate, delta, cost, a, theta_prime


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
            (a_0, z_0) = forward_propagate(self.training_data, self.theta)
            print 'cost', cost(a_0[-1], self.Y)

            test = i % self.m
            x = self.training_data[test]
            y = self.Y[test]

            (a_, z_) = forward_propagate(x, self.theta)

            gradients = theta_prime(a_, z_, self.theta, y)

            for (ix, g) in enumerate(gradients):
                self.theta[ix] -= self.l * g

        (a_final, z_final) = forward_propagate(self.training_data, self.theta)
        print 'cost', cost(a_final[-1], self.Y)

    pass


def mnist():
    from lib.mnist import read
    labels, images = read('training', path='data')
    label_values = np.unique(np.array(labels))
    result = (labels == label_values).astype(float)
    s = SimpleMachine(images, result, [784, 10])
    s.train(10)


def simplest():
    t = np.matrix(((1.0, 0, 0, 0.99), (0, 0.8, 0, 0.95), (0, 0, 0.9, 0.9),
                   (1.0, 0, 0, 0.0), (0, 0.8, 0, 0.0), (0, 0, 0.9, 0.0)))
    l = [4, 3]
    r = np.matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (1, 0, 0), (0, 1, 0), (0, 0, 1)))
    s = SimpleMachine(t, r, l)
    s.train(10000)


if __name__ == "__main__":
    #simplest()
    mnist()
