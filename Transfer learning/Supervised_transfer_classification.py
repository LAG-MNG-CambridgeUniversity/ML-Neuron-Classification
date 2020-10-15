import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

#Setting correct path 
cwd = os.getcwd() # Get current working directory
root_folder = os.sep + "ML-Neuron-Classification"
# Move to 'utils' from current directory position
sys.path.insert(0, cwd[:(cwd.index(root_folder)+len(root_folder))] + os.sep + "utils" + os.sep)

from fully_connected_ANN import*
from convolutional_neural_network import*
from helper_functions import*
from unpacking_dat_files import*


#Load data
#data1 = ReadBinWaveform('C:/Users/Marius/Documents/Datasets/.dat_testing/ec014_Spikes__Waveform.bin')
#parameters1 = ReadBinParameters('C:/Users/Marius/Documents/Datasets/.dat_testing/ec014_Spikes__Parameters.bin')
data1 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_spikes.txt', usecols=list(range(0,256)), skip_header=1)
parameters1 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_spikes.txt', dtype=None, encoding='UTF-8', usecols=list(range(256,267)), skip_header=1)

data2 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec015.03_042_044_046_spikes.txt', usecols=list(range(0,256)), skip_header=1)
parameters2 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec015.03_042_044_046_spikes.txt', dtype=None, encoding='UTF-8', usecols=list(range(256,267)), skip_header=1)

print('Initial data shapes:')
print(data1.shape)
print(parameters1.shape)
print(data2.shape)
print(parameters2.shape)

X_train = data1
Y_train = creating_y_labels_from_parameters_file(parameters1, 'ExcvsInh')

X_test = data2
Y_test = creating_y_labels_from_parameters_file(parameters2, 'ExcvsInh')

#Delte 'neither' and 'double-con' option in train and test set
X_train = X_train[Y_train[:,0] != 3]
Y_train = Y_train[Y_train[:,0] != 3]
X_train = X_train[Y_train[:,0] != 2]
Y_train = Y_train[Y_train[:,0] != 2]  

X_test = X_test[Y_test[:,0] != 3]
Y_test = Y_test[Y_test[:,0] != 3]
X_test = X_test[Y_test[:,0] != 2]
Y_test = Y_test[Y_test[:,0] != 2]  

#Introduce train-dev set to evaluate variance or data missmatch
idx = np.random.choice(X_train.shape[0], 5000, replace=False)
Y_dev_train = Y_train[idx,:]
X_dev_train = X_train[idx,:]
X_train = np.delete(X_train,idx,0)
Y_train = np.delete(Y_train,idx,0)


print('Final shapes of preprocessed data:')
print(X_train.shape)
print(Y_train.shape)
print(X_dev_train.shape)
print(Y_dev_train.shape)
print(X_test.shape)
print(Y_test.shape)

#Normalization
X_train, X_test = normalization(X_train, X_test)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l2'))
model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(10, activation='relu', kernel_regularizer='l2'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))



model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs=5)

print("The model correctly classifies ", model.evaluate(X_dev_train, Y_dev_train)[1], "%")
print("The model correctly classifies ", model.evaluate(X_test, Y_test)[1], "%")