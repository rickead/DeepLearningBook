'''

Example 5-5, pg 207 - Using Autoencoders for Anomaly Detection

Demonstrate how anomaly detection performs on the MNIST dataset using a simple autoencoder without pretraining.

Dataset: http://yann.lecun.com/exdb/mnist/


All code is written to run on Tensorflow 2 using the embedded Keras API.

'''
###############################################################################
# Imports
# Modules needed by the script.
###############################################################################
import tensorflow.python.platform
import tensorflow as tf
import numpy as np
import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D


###############################################################################
# Functions
###############################################################################
def show_batch(image_batch, label_batch, title="", display_time=5):
    plt.figure(figsize=(10,10))
    plt.title(title)
    N = len(label_batch)
    rows = N/5
    for n in range(N):
        ax = plt.subplot(rows,5,n+1)
        plt.imshow(image_batch[n])
        plt.title('Digit is {}'.format(int(label_batch[n])))
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    if display_time >= 0:
        plt.show(block=False)
        plt.pause(display_time)
    else:
        plt.show()
    plt.close()

###############################################################################
# Main program
#
# The main script follows. The if __name__ business is necessary for the
# contents of a file to be used both as a module or as a program. It is
# also necessary if you use the 'pydoc' command to generate program
# documentation.
###############################################################################
if __name__ == '__main__':

    X, y, y_onehot = mnist.load(split='train')
    X_test, y_test, y_test_onehot = mnist.load(split='test')

    # Binarization Set all of the values in the map to either 1 or a zero
    X[X >= 0.5] = 1
    X[X < 0.5] = 0
    X_test[X_test >= 0.5] = 1
    X_test[X_test < 0.5] = 0

    # Build the neural network. 784 in/out as MNIST images are 28x28
    NUM_IN = X.shape[1] * X.shape[2]
    NUM_LABELS = y_onehot.shape[1]
    NUM_NODES = 250

    # Flatten X into a batch, 784 shape
    X = X.reshape((X.shape[0], NUM_IN))
    X_test = X_test.reshape((X_test.shape[0], NUM_IN))

    # The show_batch function expects a series of grayscale images. Reshape the
    # array to provide those images.
    batch = X[0:25,].reshape(25,28,28)
    show_batch(batch, y[0:25,], "Example digits", 5)

    # Specify the same model that is used in the DL4J book.
    # 4 layers: Dense, Dense, Dense, Output
    # 784 -> 250 -> 10 -> 250 -> 784
    print("\nBuild model... {0} -> {1} -> {2} -> {1} -> {0}".format(NUM_IN, NUM_NODES, NUM_LABELS))

    # Here we will specify the "Xavier" normal initializer
    initializer = tf.keras.initializers.GlorotNormal()

    model = Sequential([
        tf.keras.layers.InputLayer(input_shape=(NUM_IN,)),
        tf.keras.layers.Dense(NUM_NODES, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(NUM_LABELS, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(NUM_NODES, activation='relu', kernel_initializer=initializer),
        # Output layer is equivalent to Dense layer + Loss layer
        # Keras doesn't have a 'loss' layer. The loss is utilized
        # when the model is compiled.
        tf.keras.layers.Dense(NUM_IN, activation='softmax', kernel_initializer=initializer)
    ])

    # The book uses a learning rate of 0.05 with the adagrad optimizer
    opt = tf.keras.optimizers.Adagrad(learning_rate=0.05)

    # The book lists stochastic gradient descent as the optimization algorithm
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    model.summary()

    NUM_EPOCHS = 30
    BATCH_SIZE = 100
    history = model.fit(x=X, y=X, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

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
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')    

    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # Evaluate the model on test data. Score each digit/example in test set separately
    # then add trip (score, digit, and INDArray data) to lists and sort by score.
    # This allows us to get best N and work N digits for each type
    print("Evaluate the resuts on the test data")
    results = model.evaluate(x=X_test,y=X_test, batch_size=BATCH_SIZE)
    print("test loss, test acc:", results)

    print("Generate predictions on the new data using predict")
    predictions = model.predict(X_test)
    print("predictions shape:", predictions.shape)
    score = tf.math.reduce_sum(predictions, 1)
    print("sum predictions shape:", score.shape)

    listsByDigit = {}
    for i in range(NUM_LABELS):
        listsByDigit[i] = list()

    # Arrange scores and their index in the test data into a map by digit
    for idx,score_at_idx in enumerate(score):
        listsByDigit[y_test[idx]].append( (score_at_idx, idx) )

    # Sort the tuples in assending order
    for i in range(NUM_LABELS):
        listsByDigit[i].sort(key=lambda score_idx_tupple: score_idx_tupple[0])
    
    # Select the best 5 and worst 5 digits and visualize them.
    bestFiveIndex = list()
    worstFiveIndex = list()
    for i in range(NUM_LABELS):
        # Get best of five for this digit
        score_idx_tupple = listsByDigit[i][-6:-1]
        for score_idx_tupple, idx in score_idx_tupple:
            bestFiveIndex.append(idx)
        # Get worst of five for this digit
        score_idx_tupple = listsByDigit[i][0:5]
        for score_idx_tupple, idx in score_idx_tupple:
            worstFiveIndex.append(idx)

    # Best 5 digits
    print("Best of five indexes...")
    indexes = np.array(bestFiveIndex, dtype=int)
    batch = X_test[indexes,].reshape(indexes.shape[0],28,28)
    show_batch(batch, y_test[indexes,], "Best Five Digits", -1)

    # Worst 5 digits
    print("Worst of five indexes...")
    indexes = np.array(worstFiveIndex, dtype=int)
    batch = X_test[indexes,].reshape(indexes.shape[0],28,28)
    show_batch(batch, y_test[indexes,], "Worst Five Digits", -1)




