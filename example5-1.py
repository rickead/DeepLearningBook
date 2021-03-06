"""

Example 5-1, pg 175 - Modeling CSV data with multilayer perceptron networks

Basic usage of the TensorFlow 2.3 framework using the embedded Keras API 
using a synthetic non-linear dataset and a multilayer perceptron network.

Dataset: https://github.com/jasonbaldridge/try-tf/tree/master/simdata

This is the same example as 'example5-1.py', but changed to be compatible
with the beta version of tensorflow for the Apple M1 processor.
https://github.com/apple/tensorflow_macos

"""
import tensorflow.python.platform
import tensorflow as tf
import numpy as np

import os
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

import tensorflow.python.platform
import tensorflow as tf
import wget  # pip3 install wget
import numpy as np
import datetime

import importlib

matplotlib_loader = importlib.find_loader("matplotlib")
PLT_FOUND = matplotlib_loader is not None
if PLT_FOUND:
    import matplotlib as pyplot


path_prefix = os.path.join("data", "example1")
filenameTrain = "saturn_data_train.csv"
filenameTest = "saturn_data_eval.csv"

localFilenameTrain = os.path.join(path_prefix, filenameTrain)
localFilenameTest = os.path.join(path_prefix, filenameTest)

# Data by Dr. Jason Baldridge (http://www.jasonbaldridge.com) to test neural network frameworks.
# Read "https://github.com/jasonbaldridge/try-tf/tree/master/simdata" and copy
# to data/example1
if (
    not os.path.isdir(path_prefix)
    or not os.path.exists(localFilenameTrain)
    or not os.path.exists(localFilenameTest)
):
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

NUM_LABELS = 2


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def get_dataset(file_path, **kwargs):
    """Extract tf.data.Dataset representations of labels and features in CSV files given data in the format of label, feat[0], feat[1]. feat[2], etc..

    Args:
        file_path (string): The path to one or more CSV files to load.

    Returns:
        tf.data.Dataset : A object that holds the (fetures, labels) data from the CSV file in batches.
    """
    # Use the 'experimental' make_csv_dataset to load the input data from the CSV file
    dataset = tf.data.experimental.make_csv_dataset(file_path, num_epochs=1, **kwargs)

    # Pack the features from a map of tensorflow data itnoa single feature vector.
    dataset = dataset.map(pack_features_vector)

    # Convert the integer lables in the dataset to one-hot encoded values.
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=NUM_LABELS)))
    if PLT_FOUND:
        pyplot.figure()
        # There are only two labels in this dataset 0 or 1
        idx = labels > 0.5
        pyplot.scatter(feat[idx, 0], feat[idx, 1], marker="+", c="#ff0000")
        idx = labels <= 0.5
        pyplot.scatter(feat[idx, 0], feat[idx, 1], marker="o", c="#00ff00")
        pyplot.show()

    return dataset


BATCH_SIZE = 50
NUM_EPOCHS = 40  # Number of epochs, full passes of the data
NUM_INPUTS = 2
NUM_OUTPUTS = 2
NUM_HIDDEN_NODES = 20
MY_SEED = 123

# Constants that specify the data to load from the .csv files.
COLUMN_NAMES = ["label", "x", "y"]
LABEL_NAME = COLUMN_NAMES[0]
LABELS = [0, 1]


# Load the training data set and test data set into batches and suffle the input data before use.
training_batches = get_dataset(
    localFilenameTrain,
    batch_size=BATCH_SIZE,
    column_names=COLUMN_NAMES,
    label_name=LABEL_NAME,
    shuffle=True,
    shuffle_seed=MY_SEED,
)

print("\nDataset element defintion:\n\t", training_batches.element_spec)

testing_batches = get_dataset(
    localFilenameTest,
    batch_size=BATCH_SIZE,
    column_names=COLUMN_NAMES,
    label_name=LABEL_NAME,
    shuffle=True,
    shuffle_seed=MY_SEED,
)

# Build the model. For this example, the model has two layers. The input layer is
# an multilayer perceptron network with an RELU activation function and the output
# layer is is a softmax activation function with a negative log likelihood loss function.
#
# The weight initializer in the Deep Learning book is Xavier and it is seeded with MY_SEED (123)
initializer = tf.keras.initializers.GlorotNormal(seed=MY_SEED)

model = Sequential(
    [
        tf.keras.layers.Dense(
            NUM_HIDDEN_NODES, activation="relu", kernel_initializer=initializer
        ),
        tf.keras.layers.Dense(
            NUM_OUTPUTS, activation="softmax", kernel_initializer=initializer
        ),
    ]
)

# Optimizer is Adam, loss function is mean squared error
model.compile(
    loss=tf.losses.MeanSquaredError(),
    optimizer=tf.optimizers.Adam(),
    metrics=["accuracy"],
)

print("\n\nFit the training data.")
t0 = datetime.datetime.now()

history = model.fit(training_batches, epochs=NUM_EPOCHS, verbose=1)
model.summary()

print("It took", (datetime.datetime.now() - t0).total_seconds(), " seconds to train the model.")

if PLT_FOUND:
    # plot history
    pyplot.plot(history.history["loss"], label="loss")
    pyplot.plot(history.history["accuracy"], label="accuracy")
    pyplot.title("Training loss and accuracy (MSE loss)")
    pyplot.legend()
    pyplot.show()

t0 = datetime.datetime.now()
# Run against the test set. Final evaluation of the model
scores = model.evaluate(testing_batches, verbose=0)
print("Test set analysis accuracy: %.2f%%" % (scores[1] * 100))

print("It took", (datetime.datetime.now() - t0).total_seconds(), " seconds to evaluate the model.")
