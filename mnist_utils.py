import os
import numpy as np
from mlxtend.data import loadlocal_mnist

training_files = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
testing_files = ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
NUM_LABELS = 10


def mnist_path():
    return os.path.join(os.getcwd(), "data", "mnist")


def download_data(path_prefix):
    files = (
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    )
    uri_prefix = "http://yann.lecun.com/exdb/mnist"
    os.mkdir(path_prefix)
    for f in files:
        command_string = "curl -o %s %s/%s" % (
            os.path.join(path_prefix, f),
            uri_prefix,
            f,
        )
        print(command_string)
        os.system(command_string)

    os.system("cd %s && gunzip t*-ubyte.gz" % path_prefix)


def read_dataset(path_prefix, images_file, labels_file):
    X_int, y = loadlocal_mnist(
        images_path=os.path.join(path_prefix, images_file),
        labels_path=os.path.join(path_prefix, labels_file),
    )
    # Convert labels to a one-hot matrix
    y_onehot = (np.arange(NUM_LABELS) == y[:, None]).astype(np.float32)
    # Convert the grayscale uint8 values to 32bit 0 - 1 values.
    X = np.array(X_int, dtype=np.float32) / 255.0

    # print("X Dimensions: %s x %s" % (X.shape[0], X.shape[1]))
    # print("y Dimensions: %s x %s" % (y_onehot.shape[0], y_onehot.shape[1]))

    # Reshape the input to be a 4D tensor with shape (batch, rows, cols, channels)
    X = X.reshape((X.shape[0], 28, 28, 1))
    # print("X Dimensions: ", X.shape)

    return X, y, y_onehot


def read_training_data(path_prefix):
    return read_dataset(path_prefix, training_files[0], training_files[1])


def read_testing_data(path_prefix):
    return read_dataset(path_prefix, testing_files[0], testing_files[1])

