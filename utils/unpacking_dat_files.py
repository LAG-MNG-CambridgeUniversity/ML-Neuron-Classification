import struct
import numpy as np

def ReadBinWaveform(fileName, input_length = 256):
    
    """
    Reads in a .dat file containing the dataset. Creates an 2d array with every training example in each row

    Arguments:
    fileName -- Path to datafile .dat
    input_length -- length of one training example, e.g. trace of 256 data points (here: default 256)

    Returns:
    ar -- 2d array containing whole data set, shape = (#training examples, input_length) 
    """
    
    with open(fileName, mode='rb') as file: 
        fileContent = file.read()

    a = struct.unpack("h" * ((len(fileContent)) // 2), fileContent) #from 8bit to 16 bit
    ar = np.reshape(a, (-1, input_length)) 
    return ar



def ReadBinParameters(fileName, number_of_parameters = 11):

    """
    Reads in a .dat file containing the parameters of a matching dataset. Creates an 2d array with all parameters for each training example

    Arguments:
    fileName -- Path to datafile .dat

    Returns:
    ArrayData -- 2d array containing parameter information for every training examplein data set, shape = (#training examples, number of parameters) 
    """

    with open(fileName, mode='rb') as file: 
        fileContent = file.read()

        dataStr = fileContent.decode("utf-8") #from bytes to string
    
    #sorting one string into an array of strings
    n = 0
    SingleElement = ''
    ArrayData = np.empty(1000000000, dtype=object) 
    for i in range(0,len(dataStr)):
        if dataStr[i] == '_':
            ArrayData[n] = SingleElement
            SingleElement = ''
            n=n+1
        else:
            SingleElement = SingleElement + dataStr[i]

    ArrayData[n] = SingleElement    
    ArrayData = list(filter(None, ArrayData))
    ArrayData = np.array(ArrayData)
    ArrayData = np.reshape(ArrayData, (-1, number_of_parameters))   
    
    return ArrayData