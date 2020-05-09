# example 5-1 Modeling CSV data with multilayer perceptron networks
import tensorflow.python.platform
import tensorflow as tf
import pandas as pd
import numpy as np

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from matplotlib import pyplot

# Extract tf.data.Dataset representations of labels and features in CSV files
# given data in the format of label, feat[0], feat[1]. feat[2], etc..
def get_dataset(file_path):
    # Use pandas to read the csv data.
    df = pd.read_csv(file_path, header=None)

    # The classification labels are in the first column of the data.
    target = df.pop(0)

    # Assuming that a value of zero is a label, the number of labels it the maximum integer in the array, plus 1
    NUM_LABELS = np.max(target) + 1

    # Convert the integer lables into a one hot encoding matrix
    labels_onehot = (np.arange(NUM_LABELS) == target[:, None]).astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((df.values, labels_onehot))

    return dataset

path_prefix = os.path.join("data", "classification-simdata")
# filenameTrain = os.path.join(path_prefix, "saturn_data_train.csv")
# filenameTest = os.path.join(path_prefix, "saturn_data_eval.csv")
filenameTrain = "saturn_data_train.csv"
filenameTest = "saturn_data_eval.csv"

# Data by Dr. Jason Baldridge (http://www.jasonbaldridge.com) to test neural network frameworks.
# Read "https://github.com/jasonbaldridge/try-tf/tree/master/simdata" and copy
# to data/classification-simdata
if not os.path.isdir(path_prefix):
     print("Missing Saturn simulation data!")
     printf("Downloading from https://github.com/jasonbaldridge/try-tf/tree/master/simdata")

tf.keras.backend.set_floatx('float64')

# Load the training data set
raw_train_data = get_dataset(os.path.join(path_prefix, filenameTrain))

print("\n\nTraining data set")
for feat, targ in raw_train_data.take(5):
    print ('Features: {}, Target: {}'.format(feat, targ))

# Load the test/evaluation data set
raw_test_data = get_dataset(os.path.join(path_prefix, filenameTest))

print("\n\nTesting data set")
for feat, targ in raw_test_data.take(5):
    print ('Features: {}, Target: {}'.format(feat, targ))

print("\n\n")

BATCH_SIZE = 50
seed = 123
LEARNING_RATE = 0.005
NUM_EPOCHS = 30 # Number of epochs, full passes of the data
NUM_INPUTS = 2
NUM_OUTPUTS = 1
NUM_HIDDEN_NODES = 20

# Build the model. For this example, the model has two layers. The input layer is 
# an multilayer perceptron network with an RELU activation function and the output
# layer is is a softmax activation function with a negative log likelihood loss function.
# 
# model = Sequential()
# model.add(Dense(NUM_HIDDEN_NODES, input_shape=(NUM_INPUTS,1), activation='relu'))
# model.add(Dense(NUM_OUTPUTS, input_shape=(NUM_INPUTS, NUM_HIDDEN_NODES), activation='softmax'))
# model.summary()

model = Sequential([
    tf.keras.layers.Dense(NUM_HIDDEN_NODES, activation='relu'),
    tf.keras.layers.Dense(NUM_OUTPUTS, activation='softmax')
])

# For this example, we need to calcualte the negative log likelihood of the model given the data
# To do this with Keras, we need to create a class that inhertis from the tf.keras.losses.Loss class 
# and implement the following two methods:
#   __init__(self) —Accept parameters to pass during the call of your loss function
#   call(self, y_true, y_pred) —Use the targets (y_true) and the model predictions (y_pred) to compute the model's loss
#
# See: 
class NegLogLikelihood(tf.keras.losses.Loss):
    """Loss class calcuates the negative log likelihood of the model, given the data.
    
    Arguments:
        model -- The Keras neural network model
        reduction -- Type of tf.keras.losses.Reduction to apply to loss.
        name -- Name of the loss function.
    """
    def __init__(self, model, reduction=tf.keras.losses.Reduction.AUTO, name='nll_gausian'):
        super().__init__(reduction=reduction, name=name)
        self.model = model

    """Need to convert the loss function below to a 
    loss function suitable to the above input parameters.

    Likelihood is the probability that the calculated parameters
    produced the known data. Probability of the parameters (model)
    given the data.

    Likelihood:
    L = Product i=1..N p(x(i) | theta)

    NLL:
    NLL = Sum i=1..N -log(p(x(i) | theta))

    where, p(x(i) | theta) is the gausian probability density function
    """
    def call(self, y_true, y_pred):
        print("y_true:", y_true, ", y_pred:", y_pred)
        y_pred_mean = tf.math.reduce_mean(y_pred, axis=-1)
        y_pred_sd = tf.math.reduce_std(y_pred, axis=-1)

        print("mean:", y_pred_mean, ", sd:", y_pred_sd)

        ## element wise square
        square = tf.square(y_pred_mean - y_true)## preserve the same shape as y_pred.shape
        ms = tf.add(tf.divide(square,y_pred_sd), tf.math.log(y_pred_sd))
        ## axis = -1 means that we take mean across the last dimension 
        ## the output keeps all but the last dimension
        ## ms = tf.reduce_mean(ms,axis=-1)
        ## return scalar
        ms = tf.reduce_mean(ms)
        return(ms)

# Optimizer is stochastic gradient descent (sgd), loss function is negative log likelihood
model.compile(optimizer='sgd', loss=NegLogLikelihood(model), metrics=['accuracy'])

history = model.fit(raw_train_data, epochs=NUM_EPOCHS, verbose=1) # batch_size=BATCH_SIZE not usabled with a pandas data frame
model.summary()

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.legend()
#pyplot.show()

# Run against the test set. Final evaluation of the model
scores = model.evaluate(raw_test_data, verbose=0)
print("Test set analysis accuracy: %.2f%%" % (scores[1]*100))
