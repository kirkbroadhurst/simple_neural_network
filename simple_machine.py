import numpy as np
from lib.network import build_synapses, forward_propagate, cost, theta_prime, softmax


class SimpleMachine(object):
    
    def __init__(self, training_data, result, layers, theta=[]):
        """
        Create a simple neural network machine for classification
        :param training_data: input data as numpy matrix, n rows of observations with m features
        :param result: observed results, n rows of a single value
        :param layers: an array where each value indicates the size of a hidden layer
        """
        self.training_data = training_data        
        self.Y = result
        self.layers = layers
        if theta:
            self.theta = theta
        else:
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

        for i in range(iterations % self.m):
            #if ((1.0 * i) % 100) == 0:
            (a_0, z_0) = forward_propagate(self.training_data, self.theta)
            print 'cost', cost(a_0[-1], self.Y)

            test = i % self.m
            x = self.training_data[test]
            y = self.Y[test]

            (a_, z_) = forward_propagate(x, self.theta)

            s = softmax(z_[-1])

            # some hacks to monitor progress
            # uncomment this to see progress
            '''
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


def mnist():
    from lib.mnist import read
    labels, images = read('training', path='data')

    # normalize the pixel values; avoid overflow
    image_values = (images-128.0)/128.0
    label_values = np.unique(np.array(labels))
    result = (labels == label_values).astype(float)

    try:
        c = np.load('model_coefficients.dat')
        s = SimpleMachine(image_values, result, [784, 10], c)
    except IOError:
        s = SimpleMachine(image_values, result, [784, 10])
    c = s.train(100000)
    np.save('model_coefficients.dat', c)


def simplest():
    t = np.matrix(((1.0, 0, 0, 0.99), (0, 0.8, 0, 0.95), (0, 0, 0.9, 0.9),
                   (1.0, 0, 0, 0.0), (0, 0.8, 0, 0.0), (0, 0, 0.9, 0.0)))
    l = [4, 3]
    r = np.matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (1, 0, 0), (0, 1, 0), (0, 0, 1)))
    s = SimpleMachine(t, r, l)
    s.train(1000)


if __name__ == "__main__":
    mnist()
