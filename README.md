# Deep Learning Book Examples for TensorFlow 2.3
This repository contains a set of examples from the Deep Learning book, written by Jash Patterson & Adam Gibson. The examples in the book were written for DL4J (https://deeplearning4j.org/) and I have translated these examples into TensorFlow 2.3 with Keras.

The final version of each example is in this top-level directory, with scripts that are not ready for publication in the 'future' directory.

# Dependencies
 - Tensor 2.6
 - python wget

 # Files
  - 'example5-1.py' - Modeling CSV data with a multilayer perceptron network. Test set accuracy 97.98%.
  - 'example5-1b.py' - Uses CategoricalCrossentropy as the loss function instead of Mean Squared Error. Gradient Descent Optimizer Test set accuracy 92.93%. Adam Optimizer Test set accuracy 94.95%.
  - 'example5-2.py' - Modeling handwritten images using CNNs. (Convolutional Neural Networks). Model validation accuracy is 98.27%.
  - 'example5-3.py' - Generating Shakespeare via LSTMs
  - 'example5-4.py' - Classifying Sensor Time-series Sequences using LSTMs
  - 'example5-5.py' - Using Autoencoders for Anomaly Detection. The resulting best five digits and worst five digits are all classified correctly.
  
  