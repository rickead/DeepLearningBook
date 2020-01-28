'''
LSTM example using 

https://archive.ics.uci.edu/ml/datasets/synthetic+control+chart+time+series

The data is stored in an ASCII file, 600 rows, 60 columns, with a single chart per line. The classes are organized as follows:

1-100   Normal
101-200 Cyclic
201-300 Increasing trend
301-400 Decreasing trend
401-500 Upward shift
501-600 Downward shift

All code is written to run on Tensorflow 2 using the embedded Keras API.

'''
# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers.embeddings import Embedding
#from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import TimeDistributed

from matplotlib import pyplot
import random

# This class causes Tensorflow to crash
class NormalizerStandardizeSeq(Sequence):
    """
    A stdiance and mean pre-processor for a data set that normalizes feature values to have
    0 mean and a standard deviation of 1.
    """
    def __init__(self, data):
        # Calculate the mean and standard deviation of the input data to use for
        # normalizing the data stream.
        self.__mean = 0.0
        self.__std = 1.0
        # For a single axis, take the mean. For two axis (batch, features) or three axis
        # (batch, sequence, features), take the mean of each feature.
        # TODO Verify that the data dimentions are the same as the x dimetions

        if data.ndim == 1:
            # For a single axis, take the mean of all of the data.
            self.__mean = np.mean(data)
            self.__std = np.std(data)
            self.__axis = 0
        elif data.ndim == 2:
            # For two axes, take the mean of each column of data, assuming the
            # second axis is the feature axis.
            self.__mean = np.mean(data, axis=1)
            self.__std = np.std(data, axis=1)
            self.__axis = 1
        elif data.ndim == 3:
            # For three axes, we want the mean of each feature, where the last axis
            # is the feature axis
            statsShape = (data.shape[0]*data.shape[1], data.shape[2])
            self.__mean = np.mean(data.reshape(statsShape))
            self.__std = np.std(data.reshape(statsShape))
            print("mean=", self.__mean)
            print(" std=", self.__std)
            self.__axis = 2

    def normalize(self, x):
        features_mean = np.full(x.shape, self.__mean)
        z = np.subtract(x, features_mean)
        features_std = np.full(x.shape, self.__std)
        z = np.divide(z, features_std)
        return z

    def attach(self, x_data, y_data, batch_size):
        """
        Attach a set of data to this normalizing sequencer. 
        """
        self.__x = x_data
        self.__y = y_data
        self.__batch_size = batch_size

    def __len__(self):
        return self.__x.shape[0]

    def __getitem__(self, idx):
        batch_x = self.normalize(self.__x[idx * self.__batch_size : (idx + 1) * self.__batch_size])
        batch_y = self.__y[idx * self.__batch_size : (idx + 1) * self.__batch_size]
        return batch_x, batch_y

class NormalizerStandardize:
    def __init__(self, data):
        statsShape = (data.shape[0]*data.shape[1], data.shape[2])
        self.__mean = np.mean(data.reshape(statsShape))
        self.__std = np.std(data.reshape(statsShape))
        print("mean=", self.__mean)
        print(" std=", self.__std)

    def normalize(self, x):
        """This function returns an array of the input data that has been normalized to a Z-score.
        
        Arguments:
            x {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        features_mean = np.full(x.shape, self.__mean)
        z = np.subtract(x, features_mean)
        features_std = np.full(x.shape, self.__std)
        z = np.divide(z, features_std)
        return z

# fix random seed for reproducibility
np.random.seed(123)

dataset = loadtxt('data/synthetic_control.data')

# Input shape is 60 timesteps with 1 feature per timestep
NUM_TIMESTEPS = 60
NUM_FEATURES = 1
NUM_CLASS_LABELS = 6
NUM_EXAMPLES = 600
# reshape the data set into rows (examples) by cols (samples in each example)
dataset = np.reshape(dataset, (NUM_EXAMPLES, NUM_TIMESTEPS, NUM_FEATURES))

# Seutp the labels for the dataset. The array shouuld be
# number examples by the number of class labels, for one-hot encoding
# examples 0 - 99 set the first column to 1 and the rest of the columns to 0.
# This is one hot encoding. Each column represents a class label.
# The class labels for the example data are:
# 1-100   Normal
# 101-200 Cyclic
# 201-300 Increasing trend
# 301-400 Decreasing trend
# 401-500 Upward shift
# 501-600 Downward shift
labels = np.zeros((NUM_EXAMPLES,NUM_CLASS_LABELS), dtype='float32')
for i in range(NUM_CLASS_LABELS):
    k = i*100
    labels[k:k+99,i] = 1.0

# plot each column
if 0:
    pyplot.figure()
    groups = 6
    for group in range(groups):
        pyplot.subplot(groups, 1, group+1)
        pyplot.plot(dataset[group+400,:])
    pyplot.title('Test')
    pyplot.show()

# Divide the data into a training set and a test set.
NUM_TRAIN = 450 # 75% training, 25% test
BATCH_SIZE = 32

# Create a set of indexes and shuffle them.
random.seed(12345)
shuffled_idx = random.sample(tuple(range(NUM_EXAMPLES)), k=NUM_EXAMPLES)

train_x, train_y = dataset[shuffled_idx[:NUM_TRAIN]], labels[shuffled_idx[:NUM_TRAIN]]
test_x, test_y = dataset[shuffled_idx[NUM_TRAIN:]], labels[shuffled_idx[NUM_TRAIN:]]

# Normalize the training and test data. Populate the statistics with the training 
# data set and use that to normalize the test data set.
normalizer = NormalizerStandardize(train_x)
train_x = normalizer.normalize(train_x)
test_x = normalizer.normalize(test_x)

# In this example, the dataset is an array of examples of time-series signals
# Each example is 60 samples of time-series data and the output is the classification label for
# the data.
#
# The input shape for the data for an LSTM is (batch_size, num_samples, num_features)
#     - batch_size - The number of examples in our dataset.
#     - num_samples - The number of samples, or time steps, of real-time data
#     - num_features - The number of features in each sample vector. In this case, we have only one feature.
# 
# The input_shape parameter is the input shape of the data, excluding the batch_size.
# In this case, the dataset is (600, 60, 1), and the input shape is (60,1)
nn_input_shape = (dataset.shape[1], dataset.shape[2])
print("nn_input_shape is ", nn_input_shape)

INTERNAL_UNITS = 20
# ------ Configure the Network -----
model = Sequential()
model.add(LSTM(INTERNAL_UNITS, input_shape=nn_input_shape, return_sequences=False))
# Original activation was 'sigmoid'. Changing to 'softmax' from book.
model.add(Dense(NUM_CLASS_LABELS, input_shape=(nn_input_shape[0], INTERNAL_UNITS), activation='sigmoid'))
model.summary()

# Think about adding precision and recall.
# Tried 'adam' optimizer and the Stocastic Gradient Descent 'sgd' optimizer.
# The 'sgd' optimizer was used in the book, but the 'adam' optimizer provides sbeter validation results.
# Loss function in book is MCXENT = Multiclass cross entropy. ts = tf.keras.losses.CategoricalCrossentropy()
# but this gave horrible results. Sticking with 'binary_crossentropy'.
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ----- Train the network, evaluating test set performance at each epoch -----
print("Dataset Shape = ", dataset.shape)
print("Lables Shape = ", labels.shape)

# default batch size of 32
nEpochs = 40
history = model.fit(train_x, train_y, epochs=nEpochs, verbose=1)
# plot history
pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Run against the test set. Final evaluation of the model
scores = model.evaluate(test_x, test_y, verbose=0)
print("Test set analysis accuracy: %.2f%%" % (scores[1]*100))
