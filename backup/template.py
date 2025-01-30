import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import keras as K
# import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
# from modules.target import TargetClassifier
# from modules.acgan import Generator, Discriminator, ACGAN

os.environ["TF_ENABLE_ONEDNN_OPTS"]= "0"

# Check the device GPU or CPU
print(tf.config.list_physical_devices('GPU'))



#=============================================================================================================================
#                                                   MAIN PROCESS
#=============================================================================================================================

if __name__ == "__main__":

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data(path="mnist.npz")

    print("Training set:", x_train.shape)
    print("Training label:", y_train.shape)
    print("Test set:", x_test.shape)
    print("Test label:", y_test.shape)

    print("Normalizing and Reshaping the data...")
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype(np.float32)
    x_train = (x_train - 127.5) / 127.5

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype(np.float32)
    x_test = (x_test - 127.5) / 127.5  # Corrected variable name to x_test

    # Hyperparameters
    latent_dim = 128 #noise size
    n_classes = 10
    batch_size = 100
    epochs = 20
    epochs_atgan = 20

    # PREPROCESS DATASET
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=x_train.shape[0])
    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(
        buffer_size=tf.data.AUTOTUNE
    )
    print("Dataset loaded with normalization.")
    #-------------------------------------------end of generation---------------------------------------------