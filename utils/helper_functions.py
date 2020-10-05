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

def normalization(Data_2d_array, alignment_training_examples = 'stacked_rows'):
  """
  Normalizes every training example of the training set in a feature-wise manner

  Arguments:
  Data_2d_array -- A 2d array containing all training examples (stack of 1d training traces)
  alignment_training_examples -- Describes in which dimensions the training examples are stacked 
                                 (default = 'stacked_rows' means each training example is a row vector 
                                 and they are stacked along the second axis, 
                                 i.e. data.shape = (#training examples, #features per training example))
  """
  if alignment_training_examples == 'stacked_rows':
      #substract mean
      means = np.mean(Data_2d_array, axis=0)
      means = means.reshape((1, Data_2d_array.shape[1]))
      Normalized_data = Data_2d_array - means

      #Normalize over variance
      element_wise_squares = np.multiply(Normalized_data,Normalized_data)
      variances = np.mean(element_wise_squares, axis=0)
      variances = variances.reshape((1, Data_2d_array.shape[1]))
      Normalized_data = Normalized_data / variances

  elif alignment_training_examples == 'stacked_columns':
      #substract mean
      means = np.mean(Data_2d_array, axis=1)
      means = means.reshape((Data_2d_array.shape[0], 1))
      Normalized_data = Data_2d_array - means

      #Normalize over variance
      element_wise_squares = np.multiply(Normalized_data,Normalized_data)
      variances = np.mean(element_wise_squares, axis=1)
      variances = variances.reshape((Data_2d_array.shape[0], 1))
      Normalized_data = Normalized_data / variances

  else:
    raise ValueError(f'Please specify the stacking of your training examples in the argument alignment_training_examples to one of the valid options: stacked_rows, stacked_columns')

  return Normalized_data, means, variances