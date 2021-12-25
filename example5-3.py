"""
Example 5-3, pg 191 - Generating Shakespeare via LSTMs

The training dataset for this example is provided by the Complete Works of
William Shakespear (http://www.gutenbarg.org/ebooks/100)

All code is written to run on Tensorflow 2 using the embedded Keras API.

The original Java code is found at http://bit.ly/2sUsPU.

This code borrows heavily from the code in an article from towardsdatascience.com

https://towardsdatascience.com/create-your-own-artificial-shakespeare-in-10-minutes-with-natural-language-processing-1fde5edc8f28

"""
###############################################################################
# Imports
# Modules needed by the script.
###############################################################################
import tensorflow.python.platform
import tensorflow as tf
import numpy as np

import os
import sys
import optparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

print("Using TensorFlow Version", tf.__version__)

###############################################################################
# Functions
###############################################################################
def process_command_line(argv):
    """
    Return a 2-tuple: (settings object, args list).
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    global LOG
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = optparse.OptionParser(
        formatter=optparse.TitledHelpFormatter(width=78), add_help_option=None
    )

    # define options here:
    parser.add_option(
        "--train",
        dest="train_model",
        default=False,
        action="store_true",
        help="Train and test model. Default is test model only.",
    )
    parser.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose output",
    )
    parser.add_option(  # customized description; put --help last
        "-h", "--help", action="help", help="Show this help message and exit."
    )

    options, args = parser.parse_args(argv)

    return options, args


def split_input_target(chunk):
    """Create a tuple of the sequnece of characters to feed into the RNN model.
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_gru_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, batch_input_shape=[batch_size, None]
            ),
            tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer='glorot_uniform',
            ),
            tf.keras.layers.Dense(vocab_size),
        ]
    )
    return model


def build_lstm_model(vocab_size, embedding_dim, lstm_units, batch_size):
    model = Sequential(
        [
            # Input layer is an embedding layer
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, batch_input_shape=[batch_size, None]
            ),
            # The first and second layers are a Graves LSTM in the book, but this doesn't exist in Keras or TF
            # This is Long Short-Term Memory Layer - Hochreiter 1997
            tf.keras.layers.LSTM(LSTM_LAYER_SIZE),  # tahn activation is the default
            tf.keras.layers.LSTM(LSTM_LAYER_SIZE),
            tf.keras.layers.Dense(vocab_size),
        ]
    )
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)
    print("------------ LOAD DATASET -----------")
    vocab_size, char2idx, idx2char, text = load_dataset()
    print("-------------------------------------\n")

    # Number of characters to generate
    num_generate = 500

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + "".join(text_generated)


def load_dataset():
    # Download the test data file
    path_to_file = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
    )
    # Read, then decode for py2 compat.
    text = open(path_to_file, "rb").read().decode(encoding="utf-8")
    # length of text is the number of characters in it
    print("Length of text: {} characters".format(len(text)))
    # Take a look at the first 250 characters in text
    print('The first 250 characters of the corpus are as follows:\n', text[:250])
    # The unique characters in the file
    vocab = sorted(set(text))
    # Length of vocabulary in characters
    vocab_size = len(vocab)
    print("{} unique characters".format(vocab_size))

    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    return (vocab_size, char2idx, idx2char, text)


# checkpoint_dir - Directory where the checkpoints will be saved
def train_model(checkpoint_dir, saved_model_path):
    # Number of units in each LSTM layer
    LSTM_LAYER_SIZE = 200
    # Batch Size to use when training
    BATCH_SIZE = 64
    # Length of each training example sequence to use.
    EXAMPLE_LENGTH = 100
    # Number of epochs to use during training
    NUM_EPOCHS = 10

    vocab_size, char2idx, idx2char, text = load_dataset()

    text_as_int = np.array([char2idx[c] for c in text])

    print("{")
    for char, _ in zip(char2idx, range(20)):
        print("  {:4s}: {:3d},".format(repr(char), char2idx[char]))
    print("  ...\n}")

    # Show how the first 13 characters from the text are mapped to integers
    print(
        "{} ---- characters mapped to int ---- > {}".format(
            repr(text[:13]), text_as_int[:13]
        )
    )

    # Divide the text into example sequences. Each input will contian EXAMPLE_LENGTH
    # characters from the text.
    examples_per_epoch = len(text) // (EXAMPLE_LENGTH + 1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    print("Dump five examples from char_dataset:")
    for i in char_dataset.take(5):
        print(idx2char[i.numpy()])

    sequences = char_dataset.batch(EXAMPLE_LENGTH + 1, drop_remainder=True)

    print("Dump five sequences: ")
    for item in sequences.take(5):
        print(repr("".join(idx2char[item.numpy()])))

    # Map the sequences into a tuple of sequences for feeding the RNN model.
    dataset = sequences.map(split_input_target)

    for input_example, target_example in dataset.take(1):
        print("Input data: ", repr("".join(idx2char[input_example.numpy()])))
        print("Target data:", repr("".join(idx2char[target_example.numpy()])))

    for i, (input_idx, target_idx) in enumerate(
        zip(input_example[:5], target_example[:5])
    ):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print(
            "  expected output: {} ({:s})".format(
                target_idx, repr(idx2char[target_idx])
            )
        )

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    # Shuffle the dataset and split it into 64 sentence batches
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print(dataset)

    # Embedding dimension. This is the the number of vectors in our character to vector lookup table.
    embedding_dim = 256
    rnn_units = 1024
    model = build_gru_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE,
    )

    # Check the shape of the output
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print("Input example batch: ", input_example_batch.shape, "# (batch_size, sequence_length)")
        print(
            "Prediction shape: ",
            example_batch_predictions.shape,
            "# (batch_size, sequence_length, vocab_size)",
        )

    model.summary()

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print(sampled_indices)

    print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

    ## Train the model
    ##-----------------------------------------------------------------------------
    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    print(
        "Prediction shape: ",
        example_batch_predictions.shape,
        " # (batch_size, sequence_length, vocab_size)",
    )
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    # Loss function used by DL4J is MCXENT (multi-class cross entropy). I assume that
    # this is the same loss function as categorical_crossentropy in Keras.
    # model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.compile(optimizer="adam", loss=loss)

    # Configure checkpoints
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True
    )

    history = model.fit(dataset, epochs=NUM_EPOCHS, callbacks=[checkpoint_callback])

    # Plot loss over epochs.
    plt.plot(history.history["loss"], label="train")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.show(block=False)
    plt.pause(10)
    plt.close()

    # The checkpoints saved in checkpoint_dir save the model state
    # so saving the model manually isn't necessary.
    # # Save the trained model for later use
    # print('Saving model...')
    # model.save(saved_model_path)
    # print('Model saved!!!')


def test_model(checkpoint_dir):

    # This is a hack! We should load the vocab_size from the dataset.
    vocab_size = 65
    embedding_dim = 256
    rnn_units = 1024

    #tf.train.latest_checkpoint(checkpoint_dir)

    model = build_gru_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=1,
    )
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    model.summary()

    generated_text = generate_text(model, start_string=u"ROMEO: ")

    print("\n==================================================================")
    print("\nGenerate text:")
    print(generated_text)


###############################################################################
# Main program
###############################################################################
if __name__ == "__main__":
    settings, args = process_command_line(sys.argv)

    checkpoint_dir = "./training_checkpoints"
    saved_model_path = "./ex3_saved_model"

    if settings.train_model:
        train_model(checkpoint_dir, saved_model_path)

    test_model(checkpoint_dir)
