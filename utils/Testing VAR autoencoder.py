#Testing VAR autoencoder

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K


from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler


from autoencoder import*
from helper_functions import*


#Conv VAE on Mnist
#(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

#vae = import_dense_VAE(2, (28,28,1))
#vae.compile(optimizer=tf.keras.optimizers.Adam())
#vae.fit(mnist_digits, epochs=30, batch_size=128)

#Dense VAE on Mnist
#(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
#mnist_digits = np.concatenate([x_train, x_test], axis=0)
#num_pixels = mnist_digits.shape[1] * mnist_digits.shape[2] # find size of one-dimensional vector

#mnist_digits = mnist_digits.reshape(mnist_digits.shape[0], num_pixels).astype('float32') # flatten training images
#mnist_digits /= 255

#vae = import_dense_VAE(2, (784,))
#vae.compile(optimizer=tf.keras.optimizers.Adam())
#vae.fit(mnist_digits, epochs=30, batch_size=128)


#Dense VAE on our data
data_path = 'C:/Users/Marius/Documents/Datasets/.txt/ec014.42_794_796_798_spikes.txt'
x = np.genfromtxt(data_path, usecols=list(range(0,256)), skip_header=1)
parameters = np.genfromtxt(data_path, dtype=None, encoding='UTF-8', usecols=list(range(256,267)), skip_header=1)
classification = creating_y_labels_from_parameters_file(parameters, 'ExcvsInh')
x = np.append(x, classification, axis=1)
x = x[x[:,256] != 2] 
x = x[x[:,256] != 3] 
excits = x[x[:,256] == 0]
inhibs = x[x[:,256] == 1]
rnd = np.random.choice(inhibs.shape[0], excits.shape[0], replace=False) #As more inhibs, choose randomly from inhibibitory to get same number of both types
inhibs = inhibs[rnd,:]
x = np.concatenate([excits, inhibs], axis=0) #back together
x = np.take(x,np.random.permutation(x.shape[0]),axis=0,out=x) #random shuffle
Y_train = x[:,256]
X_train = x[:,:-1]
X_train = normalization_train(X_train)
#scaler = MinMaxScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#print(X_train.shape)
#print(X_train[0,0:20])
#X_train_3D = X_train.reshape(X_train.shape[0], 8, 32, 1)
#X_train_3D = np.expand_dims(X_train_3D, -1).astype("float32")

number_of_repeats = 10
k_means_acc = []
Agglomerative_clustering = []

for d in range(number_of_repeats):
    def import_dense_VAE(latent_dimensions, input_shape):
        class Sampling(tf.keras.layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        latent_dim = latent_dimensions

        encoder_inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.Dense(input_shape[0], activation=tf.keras.layers.LeakyReLU(alpha=0.3))(encoder_inputs)
        x = tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        x = tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        x = tf.keras.layers.Dense(2000, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(2000, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(latent_inputs)
        x = tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        x = tf.keras.layers.Dense(500, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        decoder_outputs = tf.keras.layers.Dense(input_shape[0], activation="tanh")(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

        class VAE(tf.keras.Model):
            def __init__(self, encoder, decoder, **kwargs):
                super(VAE, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder

            def train_step(self, data):
                if isinstance(data, tuple):
                    data = data[0]
                with tf.GradientTape() as tape:
                    z_mean, z_log_var, z = encoder(data)
                    reconstruction = decoder(z)
                    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
                    reconstruction_loss = tf.reduce_mean(mse(data, reconstruction))
                    #reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, reconstruction))
                    reconstruction_loss *= input_shape[0]
                    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                    kl_loss = tf.reduce_mean(kl_loss)
                    kl_loss *= -0.5
                    total_loss = reconstruction_loss + kl_loss
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {
                    "loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                    "kl_loss": kl_loss,
                }

        vae = VAE(encoder, decoder)
        return vae, encoder

    vae, encoder = import_dense_VAE(5, (256,))
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    vae.fit(X_train, X_train, epochs=20, batch_size=256)

    kmeans = KMeans(n_clusters=2, n_init=50)
    y_pred = kmeans.fit_predict(encoder.predict(X_train)[2])
    print('k-means acc:', acc(Y_train, y_pred))
    k_means_acc = np.append(k_means_acc, acc(Y_train, y_pred))

    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    y_pred = cluster.fit_predict(encoder.predict(X_train)[2])
    print('Agglomerative_clustering acc:', acc(Y_train, y_pred))
    Agglomerative_clustering = np.append(Agglomerative_clustering, acc(Y_train, y_pred))

print(k_means_acc)
print(Agglomerative_clustering)