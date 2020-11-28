
###############################################################################
# Imports
# Modules needed by the script.
###############################################################################
import tensorflow.python.platform
import tensorflow as tf
from mlxtend.data import loadlocal_mnist
import numpy as np
import os

NUM_LABELS = 10
MNIST_IMAGE_SIZE = 28

###############################################################################
# Functions
###############################################################################
def get_dataset(path_prefix, images_file, labels_file):
    X_int, y = loadlocal_mnist(
        images_path=os.path.join(path_prefix, images_file),
        labels_path=os.path.join(path_prefix,labels_file),
    )
    # Convert labels to a one-hot matrix
    y_onehot = (np.arange(NUM_LABELS) == y[:, None]).astype(np.float32)
    # Convert the grayscale uint8 values to 32bit 0 - 1 values.
    X = np.array(X_int, dtype=np.float32) / 255.0

    # print('X Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
    # print('y Dimensions: %s x %s' % (y_onehot.shape[0], y_onehot.shape[1]))

    # Reshape the input to be a 4D tensor with shape (batch, rows, cols, channels)
    X = X.reshape((X.shape[0], MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1))
    # print('X Dimensions: ', X.shape)

    # X = tf.data.Dataset.from_tensor_slices(X)
    # y = tf.data.Dataset.from_tensor_slices(y)
    # y_onehot = tf.data.Dataset.from_tensor_slices(y_onehot)

    return X, y, y_onehot

def load(split=None, shuffle_files=False):
    training_files = ("train-images.idx3-ubyte",
    "train-labels.idx1-ubyte")
    testing_files = ("t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte")

    files = training_files + testing_files

    print("\nDownload data, if necessary...")
    path_prefix = os.path.join(os.getcwd(), "data", "mnist")
    if not os.path.isdir(path_prefix):
        print("Downloading data from MNIST dataset")
        files = (
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
        )
        uri_prefix = "http://yann.lecun.com/exdb/mnist"
        os.mkdir(path_prefix)
        for f in files:
            command_string = "curl -o %s.gz %s/%s.gz" % (os.path.join(path_prefix, f), uri_prefix, f)
            print(command_string)
            os.system(command_string)
        
        os.system('gunzip t*-ubyte.gz')

    features = np.empty((0, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1), dtype=np.float32)
    labels = np.empty(0, dtype=np.float32)
    labels_onehot = np.empty((0,10), dtype=np.float32)

    if not split or 'train' in split:
        print("\nLoad training data...")
        features, labels, labels_onehot = get_dataset(path_prefix, training_files[0], training_files[1])

    if not split or 'test' in split:
        print("\nLoad testing data...")
        X, y, y_onehot = get_dataset(path_prefix, testing_files[0], testing_files[1])

        features = np.append(features, X, axis=0)
        labels = np.append(labels, y, axis=0)
        labels_onehot = np.append(labels_onehot, y_onehot, axis=0)

    return (features, labels, labels_onehot)
    
