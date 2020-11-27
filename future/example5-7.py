'''

Example 5-7, pg 224 - Word2Vec Example in Applications of Deep Learning in Natural
                      Language Processing

BLAH

Dataset: 

Dependencies: conda install -c anaconda tensorflow-datasets

All code is written to run on Tensorflow 2 using the embedded Keras API.

conda activate tf2_env

'''
###############################################################################
# Imports
# Modules needed by the script.
###############################################################################
import os

# Disable Tensorflow debugging logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
# Install: pip install tensorflow-datasets
import tensorflow_datasets as tfds
# tfds.disable_progress_bar()

import os.path
import logging




###############################################################################
# Classes
###############################################################################

###############################################################################
# Functions
###############################################################################
def labeler(example, index):
    return example, tf.cast(index, tf.int64)

def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode, 
                                        inp=[text, label], 
                                        Tout=(tf.int64, tf.int64))

    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually: 
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

###############################################################################
# Main program
#
# The main script follows. The if __name__ business is necessary for the
# contents of a file to be used both as a module or as a program. It is
# also necessary if you use the 'pydoc' command to generate program
# documentation.
###############################################################################
if __name__ == '__main__':

    global LOG

    consolelevel = logging.DEBUG
    logger = logging.getLogger(__name__)
    logger.setLevel(consolelevel)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(consolelevel)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)

    logger.info("Tensorflow version %s" % (tf.__version__))

    # filePath = os.path.realpath('data/raw_sentences.txt')
    DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
    FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

    for name in FILE_NAMES:
        text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)
    
    parent_dir = os.path.dirname(text_dir)

    print(parent_dir)


    logger.info("Load & Vectorize Sentences...")

    labeled_data_sets = []

    for i, file_name in enumerate(FILE_NAMES):
        lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)


    # lines_dataset = tf.data.TextLineDataset(filePath)
    # labeled_data_sets = lines_dataset.map(lambda ex: labeler(ex, 0))

    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    TAKE_SIZE = 5000

    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
    
    all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

    for ex in all_labeled_data.take(5):
        print(ex)

    tokenizer = tfds.features.text.Tokenizer()

    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)

    vocab_size = len(vocabulary_set)
    vocab_size

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    #encoder = tfds.features.text.SubwordTextEncoder(vocabulary_set)
    example_text = next(iter(all_labeled_data))[0].numpy()
    print(example_text)
    encoded_example = encoder.encode(example_text)
    print(encoded_example)

    all_encoded_data = all_labeled_data.map(encode_map_fn)

    # Split the data into a training set and a test set.
    train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE)

    test_data = all_encoded_data.take(TAKE_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE)

    # Now, test_data and train_data are not collections of (example, label) 
    # pairs, but collections of batches. Each batch is a pair of (many examples, 
    # many labels) represented as arrays.
    #
    # To illustrate:
    sample_text, sample_labels = next(iter(test_data))

    print(sample_text[0])
    print(sample_labels[0])

    # Since we have introduced a new token encoding (the zero used for padding),
    # the vocabulary size has increased by one.
    vocab_size += 1


    logger.info("Building model...")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 64))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    # One or more dense layers.
    # Edit the list in the `for` line to experiment with layer sizes.
    for units in [64, 64]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))

    # Output layer. The first argument is the number of labels.
    model.add(tf.keras.layers.Dense(3))
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    logger.info("Fitting Word2Vec model...")
    model.fit(train_data, epochs=3, validation_data=test_data)

    # eval_loss, eval_acc = model.evaluate(test_data)
    # print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))

    logger.info("Writing word vectors to text file...")
