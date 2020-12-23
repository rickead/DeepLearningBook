'''

Example 5-1, pg 175 - Modeling CSV data with multilayer perceptron networks

Basic usage of the TensorFlow 2.3 framework using the embedded Keras API 
using a synthetic non-linear dataset and a multilayer perceptron network.

Dataset: https://github.com/jasonbaldridge/try-tf/tree/master/simdata

'''
import tensorflow.python.platform
import tensorflow as tf
import pandas as pd
import wget # pip3 install wget
import numpy as np

import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from matplotlib import pyplot

path_prefix = os.path.join("data", "example1")
filenameTrain = "saturn_data_train.csv"
filenameTest = "saturn_data_eval.csv"

localFilenameTrain = os.path.join(path_prefix, filenameTrain)
localFilenameTest = os.path.join(path_prefix, filenameTest)

# Data by Dr. Jason Baldridge (http://www.jasonbaldridge.com) to test neural network frameworks.
# Read "https://github.com/jasonbaldridge/try-tf/tree/master/simdata" and copy
# to data/example1
if not os.path.isdir(path_prefix) or not os.path.exists(localFilenameTrain) or not os.path.exists(localFilenameTest):
    # The actual URL for the raw data is:
    URL = "https://raw.githubusercontent.com/jasonbaldridge/try-tf/master/simdata/"
    print("Missing Saturn simulation data!")
    print("Downloading from", URL)
    os.mkdir(path_prefix)
    wget.download(URL + "/" + filenameTrain, localFilenameTrain)
    wget.download(URL + "/" + filenameTest, localFilenameTest)

print("\n\nExample 5.1 with TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("\nThe first five lines from the training data file:")
fd = open(localFilenameTrain)
for i in range(5):
    sys.stdout.write(fd.readline())
fd.close()

# Extract tf.data.Dataset representations of labels and features in CSV files
# given data in the format of label, feat[0], feat[1]. feat[2], etc..
def get_dataset(file_path, plotDataset=False):
    tf.keras.backend.set_floatx('float64')

    # The raw data from the file is easily loaded as a Pandas DataFrame
    df = pd.read_csv(file_path, header=None)

    # The first column is the column of classification labels. Peel off the column of labels as a 
    # vector of 32 bit integer values for use with the tf.one_hot() function.
    labels = df.pop(0)

    dataset_length = len(labels)

    # The remainder of the values are the features
    feat = df.values

    if plotDataset:
        pyplot.figure()
        # There are only two labels in this dataset 0 or 1
        idx = labels > 0.5
        pyplot.scatter(feat[idx, 0], feat[idx, 1], marker='+', c='#ff0000')
        idx = labels <= 0.5
        pyplot.scatter(feat[idx, 0], feat[idx, 1], marker='o', c='#00ff00')
        pyplot.show()

    # Assuming that a value of zero is a label, the number of labels it the maximum integer in the array, plus 1
    NUM_LABELS = np.max(labels) + 1

    # Convert the integer lables into a one hot encoding matrix
    labels_onehot = tf.one_hot(labels, depth=NUM_LABELS)

    # A tf.data.Dataset represents a sequence of elements, where each element consists of the data and the data label.
    # See: https://www.tensorflow.org/guide/data
    # As one-hot encoded data...
    dataset = tf.data.Dataset.from_tensor_slices((feat, labels_onehot))

    # The Dataset object is Python iterable.
    return dataset, dataset_length


# Load the training data set
raw_train_data, raw_train_data_length = get_dataset(localFilenameTrain, plotDataset=True)

print("\nDataset element defintion:\n\t", raw_train_data.element_spec)

print("\nTraining data set.")
for feat, targ in raw_train_data.take(5):
    print ('Features: {}, Target: {}'.format(feat, targ))

# Load the test/evaluation data set
raw_test_data, raw_test_data_length = get_dataset(localFilenameTest)

print("\nTesting data set.")
for feat, targ in raw_test_data.take(5):
    print ('Features: {}, Target: {}'.format(feat, targ))

print("\n\n")

BATCH_SIZE = 50
NUM_EPOCHS = 30 # Number of epochs, full passes of the data
NUM_INPUTS = 2
NUM_OUTPUTS = 2
NUM_HIDDEN_NODES = 20
MY_SEED=123

# Build the model. For this example, the model has two layers. The input layer is 
# an multilayer perceptron network with an RELU activation function and the output
# layer is is a softmax activation function with a negative log likelihood loss function.
# 
# The weight initializer in the Deep Learning book is Xavier and it is seeded with 123
initializer = tf.keras.initializers.GlorotNormal(seed=MY_SEED)

model = Sequential([
    tf.keras.layers.Dense(NUM_HIDDEN_NODES, activation='relu', kernel_initializer=initializer),
    tf.keras.layers.Dense(NUM_OUTPUTS, activation='softmax', kernel_initializer=initializer)
])

# To train using the Dataset, we should shuffle and batch the data
training_batches = raw_train_data.shuffle(raw_train_data_length, seed=MY_SEED).batch(BATCH_SIZE)

# Optimizer is Adam, loss function is mean squared error
model.compile(loss = tf.losses.MeanSquaredError(), optimizer = tf.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(training_batches, epochs=NUM_EPOCHS, verbose=1)
model.summary()

# plot history
pyplot.plot(history.history['loss'], label='loss')
pyplot.plot(history.history['accuracy'], label='accuracy')
pyplot.title('Training loss and accuracy (MSE loss)')
pyplot.legend()
pyplot.show()

# Run against the test set. Final evaluation of the model
testing_batches = raw_test_data.shuffle(raw_test_data_length, seed=MY_SEED).batch(BATCH_SIZE)
scores = model.evaluate(testing_batches, verbose=0)
print("Test set analysis accuracy: %.2f%%" % (scores[1]*100))
