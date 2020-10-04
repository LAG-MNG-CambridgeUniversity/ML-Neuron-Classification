import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from utils.fully_connected_ANN import*

data = np.loadtxt('C:/Users/Marius/Documents/Datasets/testing/Combined_Data_ML.txt', dtype=np.float, delimiter=',')
idx = np.random.choice(data.shape[0], 46769, replace=False)
test_labels = data[idx,256]
test_data = data[idx,0:256]
A = np.delete(data,idx,0)
training_labels = A[:,256]
training_data = A[:,0:256]

model = import_fully_connected_ANN(5, [256,256,100,100,10])

model.fit(training_data, training_labels, batch_size=32, epochs=5)

print("The model correctly classifies ", model.evaluate(test_data, test_labels)[1], "%")