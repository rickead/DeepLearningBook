'''
Example 5-2, pg 186 - Modeling Handwritten Images Using CNNs.

The training dataset that we'll use is the MNIST handwritten image dataset
(http://yann.lecun.com/exdb/mnist/)

All code is written to run on Tensorflow 2 using the embedded Keras API.


'''
###############################################################################
# Imports
# Modules needed by the script.
###############################################################################
import tensorflow.python.platform
import tensorflow as tf
import numpy as np

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

from mlxtend.data import loadlocal_mnist
import requests

# This script requires the mlxtend package and the MNIST database:
# conda install mlxtend --channel conda-forge
# For usage examples see: http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/

###############################################################################
# Functions
###############################################################################
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title('Digit is {}'.format(label_batch[n]))
        plt.axis('off')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

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
    X = X.reshape((X.shape[0], 28, 28, 1))
    # print('X Dimensions: ', X.shape)

    return X, y, y_onehot

###############################################################################
# Main program
#
# The main script follows. The if __name__ business is necessary for the
# contents of a file to be used both as a module or as a program. It is
# also necessary if you use the 'pydoc' command to generate program
# documentation.
###############################################################################
if __name__ == '__main__':
    # number of input channels
    NUM_CHANNELS = 1
    # number of possible outcomes
    NUM_LABELS = 10 
    # test batch size
    BATCH_SIZE = 64
    # Number of training epochs
    NUM_EPOCHS = 10

    """ Here is the original code from the book: http://bit.ly/2toXqtn


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations) // Training iterations as above
                .regularization(true).l2(0.0005)
                /*
                    Uncomment the following for learning decay and bias
                 */
                .learningRate(.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                .backprop(true).pretrain(false).build();

        /*
        Regarding the .setInputType(InputType.convolutionalFlat(28,28,1)) line: This does a few things.
        (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
            and the dense layer
        (b) Does some additional configuration validation
        (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
            layer based on the size of the previous layer (but it won't override values manually set by the user)
        InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
        For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
        MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
        row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
        */

    """
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

        

    print("\nLoad training data...")
    X, y, y_onehot = get_dataset(path_prefix, training_files[0], training_files[1])

    # The show_batch function expects a series of grayscale images. Reshape the
    # array to provide those images.
    batch = X[0:25,].reshape(25,28,28)
    show_batch(batch, y[0:25,])
    
    print("\nBuild model...")
    # Specify the same model that is used in the DL4J book.
    # 5 layers: Conv2D, MaxPooling2D, Conv2D, MaxPooling2D, Dense(RELU), Output
    model = Sequential([

        # Standard 2d convolutional neural network layer. Inputs and outputs 
        # have 4 dimensions with shape [minibatch,depthIn,heightIn,widthIn] and 
        # [minibatch,depthOut,heightOut,widthOut] respectively.
        #
        # Input (batch, rows, cols, channels, ) if channels_last
        # Output (batch, new_rows, new_cols, filters)
        tf.keras.layers.Conv2D(
            input_shape=(28,28,1),
            filters=20, 
            kernel_size=(5,5),
            activation='linear', # Identity activation function
            data_format='channels_last'
        ),

        # # Implements standard 2d spatial max pooling for CNNs.
        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2),
            data_format='channels_last'
        ),

        tf.keras.layers.Conv2D(
            filters=50, 
            kernel_size=(5,5),
            activation='linear', # Identity activation function
            data_format='channels_last'
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2,2),
            data_format='channels_last'
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(500, activation='relu'),
        # Output layer is equivalent to Dense layer + Loss layer
        # Keras doesn't have a 'loss' layer. The loss is utilized
        # when the model is compiled.
        tf.keras.layers.Dense(NUM_LABELS, activation='softmax')

    ])

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x=X, y=y_onehot, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
    model.summary()

    # Plot loss and accuracy over epochs.
    plt.subplot(2,1,1)
    plt.plot(history.history['loss'], label='train')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')    

    plt.show(block=False)
    plt.pause(30)
    plt.close()

    print("\nLoad testing data...")
    # Load the test images and verify the accuracy of the model.
    test_images, test_labels, test_labels_onehot = get_dataset(path_prefix, testing_files[0], testing_files[1])

    test_loss, test_acc = model.evaluate(test_images, test_labels_onehot, verbose=2)
    print("Model validation accuracy is ", test_acc)

