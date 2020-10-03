import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def import_fully_connected_ANN(number_of_layers, nodes):
	model = tf.keras.Sequential()

	for i in range(0,number_of_layers-1):
		model.add(tf.keras.layers.Dense(nodes[i], activation=tf.nn.relu))

	model.add(tf.keras.layers.Dense(nodes[number_of_layers-1], activation=tf.nn.softmax))
	model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

	return model