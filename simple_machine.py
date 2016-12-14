import numpy as np


class SimpleMachine(object):
    
    def __init__(self, training_data, result, layers):
        """
        Create a simple neural network machine for classification
        :param training_data: input data as numpy matrix, n rows of observations with m features
        :param result: observed results, n rows of a single value
        :param layers: an array where each value indicates the size of a hidden layer
        """
        self.training_data = training_data        
        self.result = result
        self.layers = layers
        self.z = []
        self.synapse = self.__construct_synapses()

    @property
    def size(self):
        return len(self.synapse)

    def nonlin(self, x, deriv=False):
        # sigmoid function
        if(deriv):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    def __construct_synapses(self):
        """
        Constructs a set of randomly initiated synapses corresponding to the number / size of hidden layers
        :return: Array of matrixes for the synapses
        """
        s = []
        for l in range(len(self.layers)):
            # for the first layer, the input size is the number of features in training data
            n = self.training_data.shape[1] if l == 0 else self.layers[l-1]
            m = self.layers[l]

            # add a random mean-centered synapse of size; 2 * random(0,1) - 1
            s += [2*np.random.random((n+1, m))-1]
        return s

    def __forward_propagate(self):
        """
        propagate through layers, applying synapses
        """
        self.z = []
        data = [self.training_data]
        for l in self.layers:

            layers += [self.nonlin(layers[i].dot(self.synapse[i]))]
        return layers

    def score(self, data):
        return self.__forward_propagate(data)[-1]
    
    
    def train(self, iterations=10000):
        self.iterations = iterations
        self.__construct_synapse();
        np.random.seed(1)        

        for i in range(self.iterations):
            layers = self.__forward_propagate(training_input)

            ######
            # backward propagate deltas
            ######
            
            # we need an error matrix for each synapse
            errors = [None] * self.size

            # set the last error value to the actual values minus the last layer
            errors[-1] = training_actual-layers[-1]

            # reverse iterate through layers, adjusting synapse according to error
            for i in range(self.size,0,-1):
                adjustment_vector = self.nonlin(layers[i], True) * errors[i-1]
                self.synapse[i-1] += layers[i-1].T.dot(adjustment_vector)
                errors[i-2] = adjustment_vector.dot(self.synapse[i-1].T)

            if i%1000==0:
                print errors[-1]

        print layers[-1]

if __name__ == "__main__":
    t = np.matrix(((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)))
    l = [3, 2]
    r = np.matrix(((1), (2), (3)))
    s = SimpleMachine(t, r, l)