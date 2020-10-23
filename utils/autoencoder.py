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
'''
def variational_autoencoder(dims, X, act='relu', init='glorot_uniform'):
  IN DEVELOPMENT
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], dims[-1]), mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon
  
    inputs = Input(shape=(256,))
    x = inputs
    for i in range(len(dims)-1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01), name='encoder_%d' % i)(x)

    z_mean = Dense(dims[-1])(x)
    z_log_sigma = Dense(dims[-1])(x)
    z = Lambda(sampling)([z_mean, z_log_sigma])
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = Input(shape=(dims[-1],), name='z_sampling')
    x = latent_inputs
    for i in range(len(dims)-1, -1, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, kernel_regularizer=tf.keras.regularizers.l2(0.01), name='decoder_%d' % i)(x)
    outputs = Dense(256, activation='sigmoid', kernel_initializer=init)(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    
    #Autoencoder
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss = 256
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    return vae, encoder, vae_loss
'''