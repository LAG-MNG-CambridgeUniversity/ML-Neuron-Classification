import datetime
import numpy as np
import tensorflow as tf
import keras.backend as K
import os
import sys

from time import time
from keras.layers import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


#Setting correct path 
cwd = os.getcwd() # Get current working directory
root_folder = os.sep + "ML-Neuron-Classification"
sys.path.insert(0, cwd[:(cwd.index(root_folder)+len(root_folder))] + os.sep)
from utils.helper_functions import*
from utils.unpacking_dat_files import*
from utils.autoencoder import*

nmi = normalized_mutual_info_score
ari = adjusted_rand_score
date = datetime.date.today()

#Upload data
data_path = 'C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_spikes.txt'
x = np.genfromtxt(data_path, usecols=list(range(0,256)), skip_header=1)
parameters = np.genfromtxt(data_path, dtype=None, encoding='UTF-8', usecols=list(range(256,267)), skip_header=1)

#x = ReadBinWaveform('C:/Users/Marius/Documents/Datasets/.dat_testing/ec014_Spikes__Waveform.bin')
#parameters = ReadBinParameters('C:/Users/Marius/Documents/Datasets/.dat_testing/ec014_Spikes__Parameters.bin')

#Include paramter information into main array
classification = creating_y_labels_from_parameters_file(parameters, 'ExcvsInh')
x = np.append(x, classification, axis=1)
x = x[x[:,256] != 2] #No neither
x = x[x[:,256] != 3] #No both

excits = x[x[:,256] == 0]
inhibs = x[x[:,256] == 1]
print("#Excting traces:", excits.shape[0])


rnd = np.random.choice(inhibs.shape[0], excits.shape[0], replace=False) #As more inhibs, choose randomly from inhibibitory to get same number of both types
inhibs = inhibs[rnd,:]
print("#Inhibiting traces:", inhibs.shape[0])
x = np.concatenate([excits, inhibs], axis=0) #back together
x = np.take(x,np.random.permutation(x.shape[0]),axis=0,out=x) #random shuffle
print("#Total traces:", x.shape[0])

Y_train = x[:,256]
X = x[:,:-1]

X_train = normalization_train(X)
X_train = X_train.reshape(X_train.shape[0], 8, 32)


print('Self-training with', X_train.shape[0], 'traces!')
n_clusters = len(np.unique(Y_train))
print(np.unique(Y_train))
print(n_clusters)

k_means_acc = []
final_acc = []

number_of_repeats = 15
for i in range(number_of_repeats):

    def autoencoder_LSTM_2D(dims, X, act=tf.keras.layers.LeakyReLU(alpha=0.3), init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None)):
        """
        RNN auto-encoder model, symmetric. Uses modulated 2D input data, eg. 1x256 trace reshaped to 16x16
        Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        X: Array of training data
        act: activation, not applied to Input, Hidden and Output layers
        return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        # input
        input_img = Input(shape=(X.shape[1],X.shape[2]), name='input')
        x = input_img
        # internal layers in encoder
        for i in range(len(dims)-1):
          x = LSTM(dims[i], activation=act, kernel_initializer=init, return_sequences=True, name='encoder_%d' % i)(x)

        encoded = LSTM(dims[-1], activation=act, kernel_initializer=init, name='encoder_%d' % len(dims))(x)
        x = encoded
        x = RepeatVector(X.shape[1])(x)

        for i in range(len(dims)-1, -1, -1):
            x = LSTM(dims[i], activation=act, kernel_initializer=init, return_sequences=True, name='decoder_%d' % i)(x)

        decoded = TimeDistributed(Dense(X.shape[2]))(x)
        return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

    #Hyperparameters
    dims = [X_train.shape[-1], 32, 16, i+1]
    #init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    pretrain_optimizer = 'Adam'
    pretrain_epochs = 250
    batch_size = 64
    save_dir = 'C:/Users/Marius/Documents/Model weights/Optimize_latent_dimensionality'
    name_save_process = f'/{date}_{i}_DEC_{dims[-1]}latentdim_pretrain.h5'
    name_save_final = f'/{date}_{i}_DEC_{dims[-1]}latentdim_final.h5'


    #Pre-training
    autoencoder, encoder = autoencoder_LSTM_2D(dims, X_train)
    autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
    autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=pretrain_epochs)
    autoencoder.save_weights(save_dir + name_save_process)


    #Clustering
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=clustering_layer)
    model.compile(optimizer='Adam', loss='kld')

    #Initialize Clusters using K-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, n_jobs=20)
    y_pred = kmeans.fit_predict(encoder.predict(X_train))
    y_pred_last = np.copy(y_pred)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    k_means_acc = np.append(k_means_acc, acc(Y_train, y_pred))

    #Iterative optimization process
    loss = 0
    index = 0
    maxiter = 2000
    index_array = np.arange(X_train.shape[0])
    tol = 0.001 # tolerance threshold to stop training
    q = model.predict(X_train, verbose=0)
    p = target_distribution(q)
    for ite in range(int(maxiter)):
        idx = index_array[index * batch_size: min((index+1) * batch_size, X_train.shape[0])]
        loss = model.train_on_batch(x=X_train[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= X_train.shape[0] else 0
        
    # evaluate the clustering performance
    model.save_weights(save_dir + name_save_final)
    q = model.predict(X_train, verbose=0)
    y_pred = q.argmax(1)
    acc_var = np.round(acc(Y_train, y_pred), 5)
    final_acc = np.append(final_acc, acc_var)
    model.save_weights(save_dir + name_save_final)

print('K_means accuracy', k_means_acc)
print('Final DEC accuracy', final_acc)


latentdim = np.linspace(1,15,15)
plt.plot(latentdim, k_means_acc, label='Acc K_means')
plt.plot(latentdim, final_acc, label='Acc DEC')
plt.title('Accuracy of K-means and DEC over latent dimensionality')
plt.xlabel('#dimensions latent space')
plt.legend()
plt.show()