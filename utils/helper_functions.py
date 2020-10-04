import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



def creating_y_labels_from_parameters_file(parameters, classification_task):
    """
    Fucntion to extract labels from paramters file depending on the classfication task

    Arguments:
    parameters -- parameters 2d array c
    classification_task -- Desired classification task, values: ['PyramidalvsInter', 'ExcvsInh'] 
   
    Returns:
    y -- array containing the labels for every training example, shape = (#training examples, 1)
    """

    if classification_task == 'PyramidalvsInter':
        y = np.empty(shape=(parameters.shape[0],1))
        for i in range(0, parameters.shape[0]):
              if parameters[i][6] == 'p': #pyramidal neuron
                y[i] = 0
              elif parameters[i][6] == 'i': #Interneuron
                y[i] = 1
              else:
                   y[i] = 2 #Neither pyramidal nor interneuron

    if classification_task == 'ExcvsInh':
        y = np.empty(shape=(parameters.shape[0],1))
        for i in range(0, parameters.shape[0]):
            if parameters[i][8] != '0' and parameters[i][9] == '0': #excitatory postsynaptic connection
                   y[i] = 0
            elif parameters[i][8] == '0' and parameters[i][9] != '0': #inhibitory postsynaptic connection
                   y[i] = 1
            elif parameters[i][8] != '0' and parameters[i][9] != '0': #both connection types
                   y[i] = 3
            else:
                y[i] = 2 #Neither excitatory or inhibitory postsynaptic connection

    return y

#def normalization(X):
