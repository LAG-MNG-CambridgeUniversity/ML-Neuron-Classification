import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def import_RNN(layer_types, dims, drop_outs, X, act = tf.keras.layers.LeakyReLU(alpha=0.3), optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']):
    """
    Implements a multi-layer CNN

    Arguments:
    layer_types -- List of strings describing the type of each layer (e.g. ['LSTM', 'LSTM', dropout', 'dense', 'dense'])
    dims -- List with length of layer_types giving the number of nodes for each LSTM or dense layer, filled up arbritrarily, recommended: 0 (e.g. [128,64,0,32,10])
    drop_outs -- either list with length of layer_types giving dropout coefficient for each dropout layer (e.g. [0,0,0.2,0,0]) or integer as dropout coefficient for all dropout layers
    X -- 2D array of training data
    input_shape -- Is passed as input_shape argument to first hidden layer
    optimizer -- Optimizer for training of ANN (default AdamOptimizer)
    loss -- Gives loss function for training of ANN (default sparse categorial crossentropy)
    metrics -- Gives the metric function that is used to judge the performance of the model (default accuracy)
    
    Returns:
    model -- Compiled CNN model, ready for training
    """
    number_of_layers = len(layer_types)
    model = tf.keras.Sequential()
    
    if layer_types[0] != 'LSTM':
        raise ValueError(f'The first layer of the network should be a LSTM layer!')
    model.add(tf.keras.layers.LSTM(dims[0], activation=act, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), input_shape=X.shape[1:]))

    for i in range(1,number_of_layers-1):
        if layer_types[i] == 'LSTM':
            model.add(tf.keras.layers.LSTM(dims[i], activation=act, return_sequences=True))

        if layer_types[i] == 'dense':
            if layer_types[:i-1].count('dense') == 0:
                model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(dims[i], activation=tf.nn.relu))

        if layer_types[i] == 'drop_out':
            if isinstance(drop_outs,float) == True:
                model.add(tf.keras.layers.Dropout(drop_outs))
            else:
                model.add(tf.keras.layers.Dropout(drop_outs[i]))

          
    if layer_types[-1] != 'dense':
        raise ValueError(f'The ouput layer should be a dense softmax layer!')

    model.add(tf.keras.layers.Dense(dims[number_of_layers-1], activation=tf.nn.softmax))
    model.compile(optimizer, loss, metrics)

    return model