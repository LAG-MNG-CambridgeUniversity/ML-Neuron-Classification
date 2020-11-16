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
from sklearn.cluster import KMeans, AgglomerativeClustering
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
#data_path = 'C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_spikes.txt'
#x = np.genfromtxt(data_path, usecols=list(range(0,256)), skip_header=1)
#parameters = np.genfromtxt(data_path, dtype=None, encoding='UTF-8', usecols=list(range(256,267)), skip_header=1)

x = ReadBinWaveform('C:/Users/Marius/Documents/Studium/PhD/Electrophysiology/ML Neuron Classification/.bin files from Buszaki/Data/CRCNS data/Ec016/ec016.19/Bin_ec016/ec016_Spikes__Waveform.bin')
parameters = ReadBinParameters('C:/Users/Marius/Documents/Studium/PhD/Electrophysiology/ML Neuron Classification/.bin files from Buszaki/Data/CRCNS data/Ec016/ec016.19/Bin_ec016/ec016_Spikes__Parameters.bin')

print(x.shape)
print(x[0,:])
print(parameters.shape)


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
X_train_3D = X_train.reshape(X_train.shape[0], 8, 32)


print('Self-training with', X_train.shape[0], 'traces!')
n_clusters = len(np.unique(Y_train))
print(np.unique(Y_train))
print(n_clusters)

number_of_repeats = 1
k_means_acc = []
k_means_acc_3D = []
Agglomerative_clustering = []
Agglomerative_clustering_3D = []
dims_matrix = np.empty(shape=(number_of_repeats,3))


for d in range(number_of_repeats):

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

    def autoencoder(dims, act='relu', init='glorot_uniform'):
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        n_stacks = len(dims) - 1
        # input
        input_img = Input(shape=(dims[0],), name='input')
        x = input_img
        # internal layers in encoder
        for i in range(n_stacks-1):
            x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

        # hidden layer
        encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

        x = encoded
        # internal layers in decoder
        for i in range(n_stacks-1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

        # output
        x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
        decoded = x
        return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

    #Hyperparameters
    #a = np.random.randint(low=1, high=500)
    #b = np.random.randint(low=1, high=500)
    #c = np.random.randint(low=1, high=500)
    a = 766
    b = 723
    c = 231
    print("a:", a)
    print("b:", b)
    print("c:", c)
    dims_matrix[d,0] = a
    dims_matrix[d,1] = b
    dims_matrix[d,2] = c
    dims = [256, a, b, c, 5]
    dims3D = [X_train_3D.shape[-1], a, b, c, 5]
    pretrain_optimizer = 'Adam'
    pretrain_epochs = 100
    batch_size = 128
    save_dir = 'C:/Users/Marius/Documents/Model weights/Optimal_Autoencoder'
    name_save_process = f'/{date}_1D_{a}_{b}_{c}_DEC__pretrain.h5'
    name_save_final = f'/{date}_1D_{a}_{b}_{c}_DEC_latentdim_final.h5'
    name_save_process_3D = f'/{date}_3D_{a}_{b}_{c}_DEC__pretrain.h5'
    name_save_final_3D = f'/{date}_3D_{a}_{b}_{c}_DEC_latentdim_final.h5'


    #Pre-training
    autoencoder, encoder = autoencoder(dims)
    autoencoder_3D, encoder_3D = autoencoder_LSTM_2D(dims3D, X_train_3D)
    autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
    autoencoder_3D.compile(optimizer=pretrain_optimizer, loss='mse')
    autoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=pretrain_epochs)
    autoencoder_3D.fit(X_train_3D, X_train_3D, batch_size=batch_size, epochs=pretrain_epochs)
    autoencoder.save_weights(save_dir + name_save_process)
    autoencoder_3D.save_weights(save_dir + name_save_process_3D)


    #Clustering
    #clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    #model = Model(inputs=encoder.input, outputs=clustering_layer)
    #model.compile(optimizer='Adam', loss='kld')

    #Initialize Clusters using K-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=50)
    y_pred = kmeans.fit_predict(encoder.predict(X_train))
    y_pred_3D = kmeans.fit_predict(encoder_3D.predict(X_train_3D))
    #y_pred_last = np.copy(y_pred)
    #model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    k_means_acc = np.append(k_means_acc, acc(Y_train, y_pred))
    k_means_acc_3D = np.append(k_means_acc_3D, acc(Y_train, y_pred_3D))

    #Agglomerative Clustering
    encoded = encoder.predict(X_train)
    encoded_3D = encoder_3D.predict(X_train_3D)
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    y_pred = cluster.fit_predict(encoded)
    y_pred_3D = cluster.fit_predict(encoded_3D)
    Agglomerative_clustering = np.append(Agglomerative_clustering, acc(Y_train, y_pred))
    Agglomerative_clustering_3D = np.append(Agglomerative_clustering_3D, acc(Y_train, y_pred_3D))

    #Iterative optimization process
    #loss = 0
    #index = 0
    #maxiter = 2000
    #index_array = np.arange(X_train.shape[0])
    #tol = 0.001 # tolerance threshold to stop training
    #q = model.predict(X_train, verbose=0)
    #p = target_distribution(q)
    #for ite in range(int(maxiter)):
        #idx = index_array[index * batch_size: min((index+1) * batch_size, X_train.shape[0])]
        #loss = model.train_on_batch(x=X_train[idx], y=p[idx])
        #index = index + 1 if (index + 1) * batch_size <= X_train.shape[0] else 0
        
    # evaluate the clustering performance
    #model.save_weights(save_dir + name_save_final)
    #q = model.predict(X_train, verbose=0)
    #y_pred = q.argmax(1)
    #acc_var = np.round(acc(Y_train, y_pred), 5)
    #final_acc = np.append(final_acc, acc_var)
    #model.save_weights(save_dir + name_save_final)

print("autoencoder dims:", dims_matrix)
print('K_means accuracy 1D:', k_means_acc)
print('K_means accuracy 3D:', k_means_acc_3D)
print('Agglo accuracy', Agglomerative_clustering)
print('Agglo accuracy 3D', Agglomerative_clustering_3D)


latentdim = np.linspace(1,number_of_repeats,number_of_repeats)
plt.plot(latentdim, k_means_acc, 'o', color='k', label='K_means 1D autoencoder')
plt.plot(latentdim, k_means_acc_3D, 'o', color='b', label='K_means 3D autoencoder')
plt.plot(latentdim, Agglomerative_clustering, 'o', color='r', label='Agglomerative_clustering 1D autoencoder')
plt.plot(latentdim, Agglomerative_clustering_3D, 'o', color='m', label='Agglomerative_clustering 3D autoencoder')
plt.title('Accuracy of K-means, Agg clustering for different autoencoder dimensions')
plt.xlabel('#Configuration')
plt.legend(loc="best")
plt.show()

plt.plot(dims_matrix[:,0], k_means_acc, 'o', color='k', label='K_means 1D autoencoder')
plt.plot(dims_matrix[:,0], k_means_acc_3D, 'o', color='b', label='K_means 3D autoencoder')
plt.plot(dims_matrix[:,0], Agglomerative_clustering, 'o', color='r', label='Agglomerative_clustering 1D autoencoder')
plt.plot(dims_matrix[:,0], Agglomerative_clustering_3D, 'o', color='m', label='Agglomerative_clustering 3D autoencoder')
plt.title('Accuracy of K-means, Agg clustering for 1st autoencoder dimensions')
plt.xlabel('#nodes 1st hidden layer')
plt.legend(loc="best")
plt.show()

plt.plot(dims_matrix[:,1], k_means_acc, 'o', color='k', label='K_means 1D autoencoder')
plt.plot(dims_matrix[:,1], k_means_acc_3D, 'o', color='b', label='K_means 3D autoencoder')
plt.plot(dims_matrix[:,1], Agglomerative_clustering, 'o', color='r', label='Agglomerative_clustering 1D autoencoder')
plt.plot(dims_matrix[:,1], Agglomerative_clustering_3D, 'o', color='m', label='Agglomerative_clustering 3D autoencoder')
plt.title('Accuracy of K-means, Agg clustering for 2nd autoencoder dimensions')
plt.xlabel('#nodes 2nd hidden layer')
plt.legend(loc="best")
plt.show()

plt.plot(dims_matrix[:,2], k_means_acc, 'o', color='k', label='K_means 1D autoencoder')
plt.plot(dims_matrix[:,2], k_means_acc_3D, 'o', color='b', label='K_means 3D autoencoder')
plt.plot(dims_matrix[:,2], Agglomerative_clustering, 'o', color='r', label='Agglomerative_clustering 1D autoencoder')
plt.plot(dims_matrix[:,2], Agglomerative_clustering_3D, 'o', color='m', label='Agglomerative_clustering 3D autoencoder')
plt.title('Accuracy of K-means, Agg clustering for 3rd autoencoder dimensions')
plt.xlabel('#nodes 3rd hidden layer')
plt.legend(loc="best")
plt.show()