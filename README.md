# ML-Neuron-Classification

The presented codes apply machine-learning techniques on multi-electrode array (MEA) data for classification of neurons. MEA data is experimentally easy to obtain, but recordings are significantly weaker and noisier. We designed and optimized artificial neural networks for supervised learning on connecting MEA data to neuronal characteristics. We developed a trained system that is capable of classifying neurons from MEA measurements in a range of dimensions (e.g. inhibitory or excitatory, pyramidal or interneuron) with high precision. The presented code 'supervised learning' is an example for classifying between exciatory and inhibitory neurons.\
Furthermore, we developed a novel unsupervised learning application for the classification of neurons based on experimental MEA measurements. To circumvent the dependence on labelled training data, we develop a system that can explore unsorted data. The designed neural network identifies potential classes within the dataset and derives a mapping between MEA data and desired number of classes (e.g. two classes for excitatory and inhibitory) with very high precision. The implementation is based on applying Deep Embedding for Clustering (DEC) to experimental MEA data.\
The neuron classification is based on a large-scale database (http://dx.doi.org/10.6080/K09G5JRZ) containing in vivo MEA recordings from hippocampal areas in rats undergoing different behavioural tasks. The data has been processed to sort the recorded spikes and to classify neurons based on monosynaptic interactions, wave-shapes and burstiness, which is used as ground truth for machine learning.\
The presented codes are stored as python notebooks, which can easily be used via Google's Colab Notebooks (used here) or Jupyther Notebook. For application, the respective MEA data file needs to be saved in a mounted google drive and linked analogue to the presented code. The MEA data files used can be accessed via https://drive.google.com/drive/folders/1cxoSeZhjI4mQHzHGZYyRwDypYp4ighrd?usp=sharing.
