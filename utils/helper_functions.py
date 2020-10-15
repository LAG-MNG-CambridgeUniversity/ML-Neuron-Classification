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
            if parameters[i][8] != 0 and parameters[i][9] == 0: #excitatory postsynaptic connection
                   y[i] = 0
            elif parameters[i][8] == 0 and parameters[i][9] != 0: #inhibitory postsynaptic connection
                   y[i] = 1
            elif parameters[i][8] != 0 and parameters[i][9] != 0: #both connection types
                   y[i] = 3
            else:
                y[i] = 2 #Neither excitatory or inhibitory postsynaptic connection

    return y

def normalization(Data_train, Data_dev, Data_test, epsilon = 0.001, alignment_training_examples = 'stacked_rows'):
  """
  Normalizes every training example of the training set in a feature-wise manner

  Arguments:
  Data_train -- A 2d array containing all training examples (stack of 1d training traces)
  Data_dev -- A 2d array containing all dev data examples (stack of 1d dev traces)
  Data_test -- A 2d array containing all test data examples (stack of 1d test traces)
  alignment_training_examples -- Describes in which dimensions the training examples are stacked 
                                 (default = 'stacked_rows' means each training example is a row vector 
                                 and they are stacked along the second axis, 
                                 i.e. data.shape = (#training examples, #features per training example))
  """
  if alignment_training_examples == 'stacked_rows':
      #substract mean
      means = np.mean(Data_train, axis=0)
      means = means.reshape((1, Data_train.shape[1]))
      Normalized_data_train = Data_train - means

      #Normalize over variance
      element_wise_squares = np.multiply(Normalized_data_train,Normalized_data_train)
      variances = np.mean(element_wise_squares, axis=0)
      variances = variances.reshape((1, Data_train.shape[1]))
      variances = np.sqrt(variances)
      Normalized_data_train /= (variances + epsilon)

      #Normalize data_test with same values
      Normalized_Data_dev = Data_dev - means
      Normalized_Data_dev /= (variances + epsilon)
      Normalized_Data_test = Data_test - means
      Normalized_Data_test /= (variances + epsilon)

  elif alignment_training_examples == 'stacked_columns':
      #substract mean
      means = np.mean(Data_train, axis=1)
      means = means.reshape((Data_train.shape[0], 1))
      Normalized_data_train = Data_train - means

      #Normalize over variance
      element_wise_squares = np.multiply(Normalized_data_train,Normalized_data_train)
      variances = np.mean(element_wise_squares, axis=1)
      variances = variances.reshape((Data_train.shape[0], 1))
      variances = np.sqrt(variances)
      Normalized_data_train /= (variances + epsilon)

      #Normalize data_dev and data_test with same values
      Normalized_Data_dev = Data_dev - means
      Normalized_Data_dev /= (variances + epsilon)
      Normalized_Data_test = Data_test - means
      Normalized_Data_test /= (variances + epsilon)

  else:
    raise ValueError(f'Please specify the stacking of your training examples in the argument alignment_training_examples to one of the valid options: stacked_rows, stacked_columns')

  return Normalized_data_train, Normalized_Data_dev, Normalized_Data_test


def Divide_train_dev_test(X, Y, fraction_list=[0.9, 0.05, 0.05], shuffle=False):
  """
  Divides a dataset into training, development and test subset

  Arguments:
  X -- A 2d array containing dataset
  Y -- A 2d array (shape = (#examples, 1) containing labels for each example)
  fraction_list -- list containing size percentages each subset should contain, e.g. [0.9, 0.05, 0.05] for 90% train, 5% dev & 5% test
  shuffle -- Boolean, if true do unison shuffling of X and Y 
  
  Returns:
  X_train, Y_train, X_dev, Y_dev, X_test, Y_test -- Respective 2d array sub datasets
  """
  if shuffle == True:
    assert X.shape[0] == Y.shape[0]
    p = np.random.permutation(X.shape[0])
    X = X[p,:]
    Y = Y[p,:]

  X_train = X[0:round(fraction_list[0]*X.shape[0]),:]
  Y_train = Y[0:round(fraction_list[0]*X.shape[0]),:]
  X_dev = X[round(fraction_list[0]*X.shape[0]):round((fraction_list[0] + fraction_list[1])*X.shape[0]),:]
  Y_dev = Y[round(fraction_list[0]*X.shape[0]):round((fraction_list[0] + fraction_list[1])*X.shape[0]),:]
  X_test = X[round((fraction_list[0] + fraction_list[1])*X.shape[0]):,:]
  Y_test = Y[round((fraction_list[0] + fraction_list[1])*X.shape[0]):,:]

  return X_train, Y_train, X_dev, Y_dev, X_test, Y_test