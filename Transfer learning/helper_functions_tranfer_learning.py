import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K




def add_noisy_version(X_values, Y_values, noise_magnitude):
    """
    Creates a 2X larger dataset by augmenting the data by adding random noise to the origninal and adding it back

    Parameters
    ----------
    X_values = numpy array of size n x 256
    Y_values = numpy array of size n x 1
    noise_magnitude = the maximum amount of noise as a fraction of the max value in the block
    
    Returns
    -------
    A dataset twice the size og the original with a noisy version of the data added to the original

    """
    
    noise = noise_magnitude* X_values.max() * np.random.rand(X_values.shape[0],X_values.shape[1])
    
    noisy_data = X_values * noise
    augmented_data= np.concatenate([X_values, noisy_data], axis=0)
    Y_labels= np.concatenate([Y_values, Y_values], axis=0)

    return augmented_data, Y_labels
    
    



def shuffle_data(X_values, Y_values):
    """
    Creates a shuffled version of the dataset 

    Parameters
    ----------
    X_values = numpy array of size n x 256
    Y_values = numpy array of size n x 1
     
    Returns
    -------
    Same shape and type of data where the 8 recording sides are shuffled

    """
    shuffled_values = X_values[:1,:]    #Just take the first value and delete later
    
    for block in X_values:
        block_x = [block[i:i + 32] for i in range(0, len(block), 32)]   #break into blocks of 32
        block_x = np.array(block_x) # 8 x 32
        # print(type(block_x))
        # print(block_x.shape)
        # print(block_x)
        
        shuffled_block = np.random.permutation(block_x)        #permutes the order of blocks
        # print(type(shuffled_block))
        # print(shuffled_block.shape)
        # print(shuffled_block)
        
      
        flattened= np.ndarray.flatten(shuffled_block)            #shuffles blocks
        # print(type(flattened))
        # print(flattened.shape)
        # print(flattened)
        
        shuffled_values= np.vstack((shuffled_values, flattened))
        
    return shuffled_values[1:], Y_values


