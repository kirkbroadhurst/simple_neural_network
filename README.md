# simplenet
A learning neural network, working against the MNIST data set but useful for general logistic neural networks.

To install

    pip install -e .

To use the MNIST dataset, [get it from Yann LeCun](http://yann.lecun.com/exdb/mnist/) and load data using the included mnist module

	from simplenet.lib.mnist import read      
	labels, images = read('training', path='data')
	  
Prepare the data
	  
    import numpy as np
      
	# normalize the pixel values; avoid overflow
	image_values = (images-128.0)/128.0
      
    # flatten the labels / Y values into a boolean map
	label_values = np.unique(np.array(labels))
	result = (labels == label_values).astype(float)    
      
Train the network
	
	from simplenet import SimpleMachine
      
    # [784, 10] are the layer sizes (784 pixels in input layer, 10 neurons in the hidden layer)
	s = SimpleMachine(image_values, result, [784, 10])      	
	  
	# train with 50000 iterations
	c = s.train(50000)
	print c

Then predict and measure accuracy
    
    predicted = s.score(image_values)
    m = np.amax(predicted, 1)
    actual_predicted = ((m * np.ones((1, 10))) == predicted)
      
    true_positives = np.sum(result[actual_predicted])
    print true_positives, labels.shape[0], 100 * true_positives/labels.shape[0]