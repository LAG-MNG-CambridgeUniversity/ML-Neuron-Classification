import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def import_fully_connected_ANN(number_of_layers, dims, optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']):
	"""
    Implements a fully connected ANN of several layers

    Arguments:
    number_of_layers -- Integer giving the number of hidden layers (e.g. 3)
    dims -- List of length number_of_layers giving the number of nodes in respective layer (e.g. [256,100,10])
    optimizer -- Optimizer for training of ANN (default AdamOptimizer)
    loss -- Gives loss function for training of ANN (default sparse categorial crossentropy)
    metrics -- Gives the metric function that is used to judge the performance of the model (default accuracy)
    
    Returns:
    model -- Compiled ANN model, ready for training
    """

	model = tf.keras.Sequential()
	for i in range(0,number_of_layers-1):
		model.add(tf.keras.layers.Dense(dims[i], activation=tf.nn.relu))

	model.add(tf.keras.layers.Dense(dims[number_of_layers-1], activation=tf.nn.softmax))
	model.compile(optimizer, loss, metrics)

	return model