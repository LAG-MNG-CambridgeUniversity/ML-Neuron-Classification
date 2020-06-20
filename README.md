# ML-Neuron-Classification

The presented codes apply machine-learning techniques on multi-electrode array (MEA) data for classification of neurons. MEA data is experimentally easy to obtain, but recordings are significantly weaker and noisier. We designed and optimized artificial neural networks for supervised learning on connecting MEA data to neuronal characteristics. We developed a trained system that is capable of classifying neurons from MEA measurements in a range of dimensions (e.g. inhibitory or excitatory, pyramidal or interneuron) with high precision. The presented code 'supervised learning' is an example for classifying between exciatory and inhibitory neurons.\\
Furthermore, we developed a novel unsupervised learning application for the classification of neurons based on experimental MEA measurements. To circumvent the dependence on labelled training data, we develop a system that can explore unsorted data. The designed neural network identifies potential classes within the dataset and derives a mapping between MEA data and desired number of classes (e.g. two classes for excitatory and inhibitory) with very high precision. The implementation is based on applying Deep Embedding for Clustering (DEC) to experimental MEA data.
