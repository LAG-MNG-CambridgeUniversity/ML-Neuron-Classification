import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tensorflow.keras.utils import plot_model
from IPython.display import Image

#Setting correct path 
cwd = os.getcwd() # Get current working directory
root_folder = os.sep + "ML-Neuron-Classification"
sys.path.insert(0, cwd[:(cwd.index(root_folder)+len(root_folder))] + os.sep)

from utils.Dense_ANN import*
from utils.CNN import*
from utils.RNN import*
from utils.helper_functions import*
from utils.unpacking_dat_files import*


#Import data, set relative path to the location of respective file!
data1 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_spikes.txt', usecols=list(range(0,256)), skip_header=1)
data2 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_noise.txt', usecols=list(range(0,256)), skip_header=1)
parameters1 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_spikes.txt', dtype=None, encoding='UTF-8', usecols=list(range(256,267)), skip_header=1)
parameters2 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_noise.txt', dtype=None, encoding='UTF-8', usecols=list(range(256,267)), skip_header=1)

#Extract labels from parameter files (here: Cluster-number)
#Add label to each session 1
classification = np.empty(shape=(data1.shape[0],1))
for i in range(0, data1.shape[0]):
    classification[i] = parameters1[i][5]
data1 = np.append(data1, classification, axis=1)
data1 = data1[data1[:,256] < 12]

#Add label to each session 2
classification = np.empty(shape=(data2.shape[0],1))
for i in range(0, data2.shape[0]):
   classification[i] = parameters2[i][5]
data2 = np.append(data2, classification, axis=1)


#Fuse both sets and shuffle
data = np.concatenate([data1, data2], axis=0)
data = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

X = data[:,0:256]
Y = data[:,256].reshape(data.shape[0], 1)



#Divide in train, dev & test and normalize
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = Divide_train_dev_test(X, Y, fraction_list=[0.9, 0.05, 0.05], shuffle=True)
X_train, X_dev, X_test = normalization(X_train, X_dev, X_test)


print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)
print('#clusters:', len(np.unique(Y_train)))

#Reshape to 3D (necessary for RNN and CNNs, but not dense)
#X_train = X_train.reshape(X_train.shape[0], 8, 32)
#X_dev = X_dev.reshape(X_dev.shape[0], 8, 32)
#X_test = X_test.reshape(X_test.shape[0], 8, 32)


#Define and compile your artificial neural network with the import functions for RNNs, CNNs or a simple dense network:
#RNN:
#model = import_RNN(layer_types=['LSTM', 'dropout', 'LSTM', 'dropout', 'LSTM', 'dense', 'dropout', 'dense'],
				   #dims=[128,0,64,0,32,32,0, len(np.unique(Y_train))],
				   #drop_outs=0.2,
				   #X=X_train)

#CNN:
'''
model = import_CNN(layer_types=['conv', 'drop_out', 'conv', 'drop_out','dense', 'dense'],
               dims= [0,0,0,0,512,10],
               filters=[64,0,16,0,0],
               kernel_sizes=[16,0,4,0,0],
               drop_outs=[0,0.75,0,0.2,0,0],
               pool_sizes=[],
               input_shape = (256,1),
               optimizer = 'adam', 
               loss = 'sparse_categorical_crossentropy', 
               metrics = ['accuracy'])
'''

#Dense network:
model = import_fully_connected_ANN(dims = [X_train.shape[-1], 500, 500, 2000, len(np.unique(Y_train))])

#Save a plot of the model model
plot_model(model, to_file='model_RNN.png', show_shapes=True)
Image(filename='model_RNN.png')

#Fit the data
history = model.fit(X_train, Y_train, epochs=20, batch_size=128)

#Evaluate the neural network on development and test set
print("RNN Model performance on \n")
print("Dev_set:", model.evaluate(X_dev, Y_dev)[1], "% \n")
print("Test_set:", model.evaluate(X_test, Y_test)[1], "%")


#plot accuracy and loss over epochs
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.title('Loss and Accuracy RNN')
plt.legend()
plt.show()