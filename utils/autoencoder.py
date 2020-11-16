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
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


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

def autoencoder_LSTM(dims, X, act=tf.keras.layers.LeakyReLU(alpha=0.3), init='glorot_uniform'):
  """
    1D RNN auto-encoder model, symmetric. Uses original 1D trace input
    Arguments:
    dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
          The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
    X: Array of training data
           act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
  """
  #n_stacks = len(dims) - 1
  length = X.shape[1]
  # input
  input_img = Input(shape=(length,1), name='input')
  x = input_img
  x = LSTM(dims[0], activation=act, kernel_initializer=init, return_sequences=True, name='encoder_0')(x)
  #x = LSTM(dims[1], activation=act, kernel_initializer=init, return_sequences=True, name='encoder_1')(x)
  #x = LSTM(dims[2], activation=act, kernel_initializer=init, return_sequences=True, name='encoder_2')(x)
  x = LSTM(dims[1], activation=act, kernel_initializer=init, return_sequences=True, name='encoder_3')(x)
  encoded = LSTM(dims[2], activation=act, kernel_initializer=init, name='encoder_4')(x)
  x = encoded
  x = RepeatVector(n=length)(x)
  #x = LSTM(dims[4], activation=act, kernel_initializer=init, return_sequences=True, name='decoder_4')(x)
  #x = LSTM(dims[3], activation=act, kernel_initializer=init, return_sequences=True, name='decoder_3')(x)
  x = LSTM(dims[2], activation=act, kernel_initializer=init, return_sequences=True, name='decoder_2')(x)
  x = LSTM(dims[1], activation=act, kernel_initializer=init, return_sequences=True, name='decoder_1')(x)
  x = LSTM(dims[0], activation=act, kernel_initializer=init, return_sequences=True, name='decoder_0')(x)
  x = TimeDistributed(Dense(1))(x)
  decoded = x
  return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


def import_ConV_VAE(latent_dimensions, input_shape):
  class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

  latent_dim = latent_dimensions

  encoder_inputs = tf.keras.Input(input_shape)
  x = tf.keras.layers.Conv2D(32, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3), strides=2, padding="same")(encoder_inputs)
  x = tf.keras.layers.Conv2D(64, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3), strides=2, padding="same")(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
  z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


  latent_inputs = tf.keras.Input(shape=(latent_dim,))
  x = tf.keras.layers.Dense(7 * 7 * 64, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(latent_inputs)
  x = tf.keras.layers.Reshape((7, 7, 64))(x)
  x = tf.keras.layers.Conv2DTranspose(64, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3), strides=2, padding="same")(x)
  x = tf.keras.layers.Conv2DTranspose(32, 3, activation=tf.keras.layers.LeakyReLU(alpha=0.3), strides=2, padding="same")(x)
  decoder_outputs = tf.keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
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
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, reconstruction))
            reconstruction_loss *= input_shape[0] * input_shape[1]
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
  return vae


def import_dense_VAE(latent_dimensions, input_shape):
  class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

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
  x = tf.keras.layers.Dense(input_shape[0], activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
  decoder_outputs = tf.keras.layers.Dense(input_shape[0], activation="sigmoid")(x)
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
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(data, reconstruction))
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
