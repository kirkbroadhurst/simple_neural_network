import numpy as np


class SimpleMachine():
    
    def __init__(self, training_data, result):
        self.training_data = training_data        
        self.result = result
        self.synapse = []
        _, self.features = training_data.shape
        

    @property
    def size(self):
        return len(self.synapse)
    
    
    def nonlin(self, x, deriv=False):
        # sigmoid function
        if(deriv):
            return x*(1-x)
        return 1/(1+np.exp(-x))            
    
        
    def __construct_synapse(self):
        # add one layer to the net for each feature        
        j = self.features
        while j > 1:
            # add a random mean-centered synapse of size n,n-1 
            self.synapse += [2*np.random.random((j,j-1))-1]
            j = j - 1
            
    
    def __forward_propagate(self, data):
        ######
        # propogate through layers, applying synapse
        ######        
        layers = [data]
        for i in range(self.size):            
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