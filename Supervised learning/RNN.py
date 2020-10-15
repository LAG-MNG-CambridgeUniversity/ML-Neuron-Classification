import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
import os
import sys

#Setting correct path 
cwd = os.getcwd() # Get current working directory
root_folder = os.sep + "ML-Neuron-Classification"
sys.path.insert(0, cwd[:(cwd.index(root_folder)+len(root_folder))] + os.sep)

from utils.fully_connected_ANN import*
from utils.convolutional_neural_network import*
from utils.helper_functions import*
from utils.unpacking_dat_files import*

#data = ReadBinWaveform('C:/Users/Marius/Documents/Datasets/.dat_testing/ec014_Spikes__Waveform.bin')
#parameters = ReadBinParameters('C:/Users/Marius/Documents/Datasets/.dat_testing/ec014_Spikes__Parameters.bin')

data1 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_spikes.txt', usecols=list(range(0,256)), skip_header=1)
data2 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_noise.txt', usecols=list(range(0,256)), skip_header=1)
parameters1 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_spikes.txt', dtype=None, encoding='UTF-8', usecols=list(range(256,267)), skip_header=1)
parameters2 = np.genfromtxt('C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_noise.txt', dtype=None, encoding='UTF-8', usecols=list(range(256,267)), skip_header=1)

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

data = np.concatenate([data1, data2], axis=0)
#Shuffle data
data = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

X = data[:,0:256]
Y = data[:,256].reshape(data.shape[0], 1)

print(X.shape)
print(Y.shape)

'''
#Create y labels from Paramters file
Y = np.empty(shape=(parameters.shape[0],1))
Y[:,0] = parameters[:,5]
X = data[Y[:,0] < 10]
Y = Y[Y[:,0] < 10] 
Y -= 2 #bias to 0
print('shape X data:', X.shape, '\n')
print('shape Y data:', Y.shape)
print(np.unique(Y))
print(len(np.unique(Y)))
'''

#Divide in train, dev & test and normalize
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = Divide_train_dev_test(X, Y, fraction_list=[0.9, 0.05, 0.05], shuffle=True)
X_train, X_dev, X_test = normalization(X_train, X_dev, X_test)
#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_dev = scaler.transform(X_dev)
#X_test = scaler.transform(X_test)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


X_train = X_train.reshape(X_train.shape[0], 16, 16)
X_dev = X_dev.reshape(X_dev.shape[0], 16, 16)
X_test = X_test.reshape(X_test.shape[0], 16, 16)


model_RNN = tf.keras.Sequential()
model_RNN.add(tf.keras.layers.LSTM(128, input_shape=(X_train.shape[1:]), activation='relu',kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None), return_sequences=True))
model_RNN.add(tf.keras.layers.Dropout(0.2))
model_RNN.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True))
model_RNN.add(tf.keras.layers.Dropout(0.2))
model_RNN.add(tf.keras.layers.LSTM(32, activation='relu', return_sequences=False))
model_RNN.add(tf.keras.layers.Flatten())
model_RNN.add(tf.keras.layers.Dense(32, activation='relu'))
model_RNN.add(tf.keras.layers.Dropout(0.2))
model_RNN.add(tf.keras.layers.Dense(len(np.unique(Y_train)), activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1.e-6)
model_RNN.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model_RNN.fit(X_train, Y_train, epochs=20, batch_size=128)

print("RNN Model performance on \n")
print("Dev_set:", model_RNN.evaluate(X_dev, Y_dev)[1], "% \n")
print("Test_set:", model_RNN.evaluate(X_test, Y_test)[1], "%")



plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Loss and Accuracy RNN')
plt.show()