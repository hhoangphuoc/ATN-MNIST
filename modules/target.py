import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import keras as K
import tensorflow as tf
import numpy as np

# =================================================================================================

@K.saving.register_keras_serializable()
class TargetClassifier(K.Model):
    """
    Target Classifier for the AT-GAN model.
    This simply acts as the classifier for the input images (MNIST) of either real or generated images.
    Using as the target for the attack.
    Architecture:
        - Conv2D(32, (3, 3), activation='relu', padding='same')
        - MaxPooling2D((2, 2))
        - Conv2D(64, (3, 3), activation='relu', padding='same')
        - MaxPooling2D((2, 2))
        - Flatten()
        - Dense(128, activation='relu')
        - Dense(num_classes, activation='softmax')
    """

    def __init__(self, num_classes=10, name="f_target", **kwargs):
        super(TargetClassifier, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes

        # Classifier Layers
        self.conv1 = K.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))
        self.pool1 = K.layers.MaxPooling2D((2, 2))
        self.conv2 = K.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = K.layers.MaxPooling2D((2, 2))

        self.flatten = K.layers.Flatten()
        self.fc1 = K.layers.Dense(128, activation='relu')
        self.fc2 = K.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def get_config(self):
        config = super(TargetClassifier, self).get_config()
        config.update({"num_classes": self.num_classes})  # Include num_classes in config
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# =======================================================================================================================

@K.saving.register_keras_serializable()
class CNNVariation(K.Model):
    """
    Variation of a CNN model with similar complexity to the TargetClassifier:
    Architecture:
        - Conv2D(32, (3, 3), activation='relu', padding='same')
        - MaxPooling2D((2, 2))
        - Conv2D(64, (5, 5), activation='relu', padding='same')
        - MaxPooling2D((2, 2))
        - Flatten()
        - Dense(1024, activation='relu')
        - Dropout(0.2)
        - Dense(num_classes, activation='softmax')
    """

    def __init__(self, name="cnn_variation", **kwargs):
        super(CNNVariation, self).__init__(name=name, **kwargs)

        # Define layers with inline configuration
        self.conv1 = K.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
        self.pool1 = K.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2 = K.layers.Conv2D(
            64, kernel_size=(5, 5), strides=1, padding="same", activation="relu"
        )
        self.pool2 = K.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = K.layers.Flatten()
        self.hidden = K.layers.Dense(1024, activation="relu")
        self.dropout = K.layers.Dropout(0.2)
        self.outputs = K.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.dropout(x, training=training)
        x = self.outputs(x)
        return x

    def get_config(self):
        config = super(CNNVariation, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build_model(self):
        inputs = tf.keras.Input(shape=(28, 28, 1))
        outputs = self.call(inputs)  # Use the defined layers
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model


@K.saving.register_keras_serializable()
class DefenceCNN(K.Model):
    """
    CNN based model for defense.
    """
    def __init__(self, name="defence_cnn", **kwargs):
        super(DefenceCNN, self).__init__(name=name, **kwargs)

        # Define layers with inline configuration
        self.conv1 = K.layers.Conv2D(
            32,
            kernel_size=(5, 5),
            strides=1,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
        self.pool1 = K.layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2 = K.layers.Conv2D(
            64, kernel_size=(5, 5), strides=1, padding="same", activation="relu"
        )
        self.pool2 = K.layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = K.layers.Flatten()
        self.fc1 = K.layers.Dense(1024, activation="relu")
        self.dropout = K.layers.Dropout(0.2)
        self.fc2 = K.layers.Dense(10, activation="softmax")  # Output layer

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x

    def get_config(self):
        config = super(DefenceCNN, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build_model(self):
        inputs = K.layers.Input(shape=(28, 28, 1))
        outputs = self.call(inputs)
        model = K.models.Model(inputs=inputs, outputs=outputs, name=self.name)
        return model
        
# =======================================================================================================================

if __name__ == "__main__":
    # =================================================================================================================
    #                             Load and Preprocess Dataset (SIMILAR TO ALL OTHER MODELS)
    # =================================================================================================================
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
    #################################################################################################################
    #================================================================================================================


    #-----------------------------------------------------------------------------------------------------------------
    #                                           Train Target Classifier
    #-----------------------------------------------------------------------------------------------------------------
    # target_classifier = TargetClassifier(num_classes=10)
    # target_classifier = CNNVariation() # CNN variation
    defence_cnn = DefenceCNN()  # Defence CNN
    target_classifier = defence_cnn.build_model()

    target_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    target_classifier.fit(x_train, y_train, epochs=20, batch_size=batch_size)

    test_loss, test_acc = target_classifier.evaluate(test_dataset)

    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")
    print("Summary: ")
    target_classifier.summary()

    # Save the target_classifer
    # target_classifier.save("..models/target/f_target.keras")
    target_classifier.save("../models/target/defence_cnn.keras")
    #================================================================================================================
