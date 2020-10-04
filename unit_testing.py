import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from utils.fully_connected_ANN import*
from utils.convolutional_neural_network import*

data = np.loadtxt('C:/Users/Marius/Documents/Datasets/testing/Combined_Data_ML.txt', dtype=np.float, delimiter=',')
idx = np.random.choice(data.shape[0], 46769, replace=False)
test_labels = data[idx,256]
test_data = data[idx,0:256]
A = np.delete(data,idx,0)
training_labels = A[:,256]
training_data = A[:,0:256]


#Reshape Training and test data for Conv network
training_data = training_data.reshape(training_data.shape[0], training_data.shape[1],1)
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1],1)

#model = import_fully_connected_ANN(5, [256,256,100,100,10]) #Testing fully connected function
model = import_CNN(layer_types=['conv', 'dense', 'dense'],
               dims= [0,512,10],
               filters=[256,0,0],
               kernel_sizes=[16,0,0],
               drop_outs=[],
               pool_sizes=[],
               input_shape = (256,1),
               optimizer = 'adam', 
               loss = 'sparse_categorical_crossentropy', 
               metrics = ['accuracy'])

model.fit(training_data, training_labels, batch_size=32, epochs=5)

print("The model correctly classifies ", model.evaluate(test_data, test_labels)[1], "%")