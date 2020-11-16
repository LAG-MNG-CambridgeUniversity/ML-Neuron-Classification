import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from matplotlib import rcParams



from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


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

def isolate_maximum_electrode(X, length=32):
  X_maxelec = np.empty(shape=(X.shape[0], length))
  for i in range(X.shape[0]):
    maxind = np.argmax(X[i,:])
    minind = np.argmin(X[i,:])
    if X[i,maxind] >= np.absolute(X[i,minind]):
      idx = maxind
    else:
      idx = minind
    elec = int(idx / 32)
    X_maxelec[i,:] = X[i,elec*32:(elec+1)*32]
  return X_maxelec

def normalization(Data_train, Data_dev, Data_test, epsilon = 0.001, alignment_training_examples = 'stacked_rows'):
  """
  Normalizes every example in the training, dev and test set in a feature-wise manner

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
      Normalized_data_train /= (variances + epsilon)

      #Normalize data_dev and data_test with same values
      Normalized_Data_dev = Data_dev - means
      Normalized_Data_dev /= (variances + epsilon)
      Normalized_Data_test = Data_test - means
      Normalized_Data_test /= (variances + epsilon)

  else:
    raise ValueError(f'Please specify the stacking of your training examples in the argument alignment_training_examples to one of the valid options: stacked_rows, stacked_columns')

  return Normalized_data_train, Normalized_Data_dev, Normalized_Data_test

def normalization_train(Data_train, epsilon = 0.001, alignment_training_examples = 'stacked_rows'):
  """
  Normalizes every training example of only the training set in a feature-wise manner

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
      Normalized_data_train /= (variances + epsilon)

  elif alignment_training_examples == 'stacked_columns':
      #substract mean
      means = np.mean(Data_train, axis=1)
      means = means.reshape((Data_train.shape[0], 1))
      Normalized_data_train = Data_train - means

      #Normalize over variance
      element_wise_squares = np.multiply(Normalized_data_train,Normalized_data_train)
      variances = np.mean(element_wise_squares, axis=1)
      variances = variances.reshape((Data_train.shape[0], 1))
      Normalized_data_train /= (variances + epsilon)

  else:
    raise ValueError(f'Please specify the stacking of your training examples in the argument alignment_training_examples to one of the valid options: stacked_rows, stacked_columns')

  return Normalized_data_train

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

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def target_distribution(q):
  """
  computing an auxiliary target distribution 
  """
  weight = q ** 2 / q.sum(0)
  return (weight.T / weight.sum(1)).T

def standardise_plot(ax, title="", xlabel="", ylabel=""):
  """Formats the plot to be ready for the final report.
    
    Args:
      ax: matplotlib axis object for a particular plot
      title (str): the title of the plot
      xlabel (str): label on the x-axis
      ylabel (str): label on the y-axis
  """
  rcParams['font.family'] = 'sans-serif'
  rcParams['font.sans-serif'] = ['Verdana']
  ax.spines["top"].set_visible(False)   # Hide top line
  ax.spines["right"].set_visible(False) # Hide right line
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  ax.ticklabel_format(useOffset=False)
  ax.set_title(f"{title}", fontsize=18)
  ax.set_xlabel(f"{xlabel}", fontsize=14)
  ax.set_ylabel(f"{ylabel}", fontsize=14)
  ax.set_xlim(0)
  #ax.set_ylim(0)
  plt.yticks(fontsize=14)
  plt.xticks(fontsize=14)
  ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
  ax.tick_params(axis = "y", which = "both", left = False, right = False)
  plt.grid(color=(0.9, 0.9, 0.9),linestyle="-", linewidth=1)

def standard_plot(X, Y, xlabel, ylabel, legend_labels, title="", labels_in_legend=True, figure_size=(6.4, 4.8), logy=False):
  """
  Args:
  places_to_plot (list): list of place ids defining which token counts should be plotted
  labels_in_legend (bool): whether to use the place labels in the legend instead of the place ids
   
  Returns: 
  """
  #Tableau of standard colors
  tableau20 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),(148, 103, 189), 
  (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
  
  # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
  for i in range(len(tableau20)):
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)
  fig, ax = plt.subplots(1, 1, figsize=figure_size)

  if Y.shape[1]>1:
    for i in range(Y.shape[1]):
      ax.plot(X, Y[:,i], 'o', color=tableau20[0], label=legend_labels[i])

  else:
    ax.plot(X, Y, 'o', color=tableau20[0], label=legend_labels[0])



  standardise_plot(ax, title , xlabel, ylabel)
  if logy:
    plt.yscale('log')
  plt.legend(fontsize=14)
  plt.title(title)
  plt.show()
  
  return fig, ax