# example 5-1 Modeling CSV data with multilayer perceptron networks
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5, # Artificially small to make examples easier to show.
#        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True, 
        **kwargs)
    return dataset

path_prefix = "data/classification-simdata/"
filenameTrain = path_prefix + "saturn_data_train.csv"
filenameTest = path_prefix + "saturn_data_eval.csv"

# Data by Dr. Jason Baldridge (http://www.jasonbaldridge.com) to test neural network frameworks.
# Read "https://github.com/jasonbaldridge/try-tf/tree/master/simdata" and copy
# to data/classification-simdata
if not os.path.isdir(path_prefix):
    print("Missing Saturn simulation data!")
    printf("Download from https://github.com/jasonbaldridge/try-tf/tree/master/simdata")
    exit(0)

# Load the training data set
raw_train_data = get_dataset(filenameTrain)

# Load the test/evaluation data set
raw_test_data = get_dataset(filenameTest)

BATCH_SIZE = 50
seed = 123
LEARNING_RATE = 0.005
NUM_EPOCHS = 30 # Number of epochs, full passes of the data
NUM_INPUTS = 2
NUM_OUTPUTS = 2
NUM_HIDDEN_NODES = 20

# Build the model. For this example, the model has two layers. The input layer is 
# an multilayer perceptron network with an RELU activation function and the output
# layer is is a softmax activation function with a negative log likelihood loss function.
# 

model = Sequential()
model.add(Dense(NUM_HIDDEN_NODES, input_shape=(NUM_INPUTS,1), activation='relu'))
model.add(Dense(NUM_OUTPUTS, input_shape=(NUM_INPUTS, NUM_HIDDEN_NODES), activation='softmax'))
model.summary()

# Example loss function for Keras
# def mean_squared_error(y_true, y_pred):
#     if not tf.is_tensor(y_pred):
#         y_pred = K.constant(y_pred)
#     y_true = K.cast(y_true, y_pred.dtype)
#     return K.mean(K.square(y_pred - y_true), axis=-1)

def nll_gaussian(y_true, y_pred):
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

    y_pred_mean = tf.math.reduce_mean(y_pred, axis=-1)
    y_pred_sd = tf.math.reduce_std(y_pred, axis=-1)

    ## element wise square
    square = tf.square(y_pred_mean - y_true)## preserve the same shape as y_pred.shape
    ms = tf.add(tf.divide(square,y_pred_sd), tf.log(y_pred_sd))
    ## axis = -1 means that we take mean across the last dimension 
    ## the output keeps all but the last dimension
    ## ms = tf.reduce_mean(ms,axis=-1)
    ## return scalar
    ms = tf.reduce_mean(ms)
    return(ms)

# Optimizer is stochastic gradient descent (sgd), loss function is 
model.compile(optimizer='sgd', loss=nll_gaussian, metrics=['accuracy'])