import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from utils.fully_connected_ANN import*
from utils.convolutional_neural_network import*
from utils.helper_functions import*
from utils.unpacking_dat_files import*

data = np.loadtxt('C:/Users/Marius/Documents/Datasets/testing/Combined_Data_ML.txt', dtype=np.float, delimiter=',')
idx = np.random.choice(data.shape[0], 46769, replace=False)
Y_test = data[idx,256]
X_test = data[idx,0:256]
A = np.delete(data,idx,0)
Y_train = A[:,256]
X_train = A[:,0:256]

#Normalization
X_train, X_test = normalization(X_train, X_test)

#Reshape Training and test data for Conv network
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)

#model = import_fully_connected_ANN(5, [256,256,100,100,10]) #Testing fully connected function
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

model.fit(X_train, Y_train, batch_size=32, epochs=5)

print("The model correctly classifies ", model.evaluate(X_test, Y_test)[1], "%")