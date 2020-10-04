import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def import_CNN(layer_types, dims, filters, kernel_sizes, drop_outs, pool_sizes, input_shape = (256,1), optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']):
    """
    Implements a multi-layer CNN

    Arguments:
    number_of_layers -- Integer giving the number of hidden layers (e.g. 3)
    dims -- List of length number_of_layers giving the number of nodes in respective layer (e.g. [256,100,10])
    optimizer -- Optimizer for training of ANN (default AdamOptimizer)
    loss -- Gives loss function for training of ANN (default sparse categorial crossentropy)
    metrics -- Gives the metric function that is used to judge the performance of the model (default accuracy)
    
    Returns:
    model -- Compiled ANN model, ready for training
    """
    number_of_layers = len(layer_types)
    model = tf.keras.Sequential()
    
    if layer_types[0] != 'conv':
        raise ValueError(f'The first layer of the network should be a convolutional layer!')
    model.add(tf.keras.layers.Conv1D(filters[0], kernel_sizes[0], activation='relu', input_shape=input_shape))

    for i in range(1,number_of_layers-1):
        if layer_types[i] == 'conv':
            model.add(tf.keras.layers.Conv1D(filters[i], kernel_sizes[i], activation='relu'))

        if layer_types[i] == 'dense':
            if layer_types[:i-1].count('dense') == 0:
                model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(dims[i], activation=tf.nn.relu))

        if layer_types[i] == 'drop_out':
            if isinstance(drop_outs,int) == True:
                model.add(tf.keras.layers.Dropout(drop_outs))
            else:
                model.add(tf.keras.layers.Dropout(drop_outs[i]))

        if layer_types[i] == 'max_pooling':
            if isinstance(pool_sizes,int) == True:
                model.add(tf.keras.layers.MaxPooling1D(pool_sizes))
            else:
                model.add(tf.keras.layers.MaxPooling1D(pool_sizes[i]))
            
    if layer_types[-1] != 'dense':
        raise ValueError(f'The ouput layer should be a dense softmax layer!')

    model.add(tf.keras.layers.Dense(dims[number_of_layers-1], activation=tf.nn.softmax))
    model.compile(optimizer, loss, metrics)

    return model