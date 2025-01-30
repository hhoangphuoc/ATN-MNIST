import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import keras as K
import numpy as np

#=====================================================================================================================
@K.saving.register_keras_serializable()
class Generator(K.Model):
    """
    Generator component of AC-GAN for MNIST dataset

    Args:
    - latent_dim: Dimension of the latent space (generated as noise)
    - n_classes: Number of classes(labels) in the dataset (default=10)

    """
    def __init__(self, latent_dim, n_classes=10, name="generator", **kwargs):
        super(Generator, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Layers for Latent Inputs
        self.dense1 = K.layers.Dense(units=7 * 7 * 256, use_bias=False)
        self.bn1 = K.layers.BatchNormalization()
        self.reshape1 = K.layers.Reshape(target_shape=[7, 7, 256])

        # Layers for Label Inputs
        self.embedding = K.layers.Embedding(input_dim=n_classes, output_dim=64)
        self.dense2 = K.layers.Dense(units=7*7, use_bias=False)
        self.bn2 = K.layers.BatchNormalization()
        self.reshape2 = K.layers.Reshape(target_shape=(7, 7, 1))

        # Layers for Merging Inputs (Combining Latent and Label Inputs)
        self.conv1 = K.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=1, padding='same', use_bias=False)
        self.bn3 = K.layers.BatchNormalization()
        self.dropout1 = K.layers.Dropout(rate=0.4)
        self.conv2 = K.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.bn4 = K.layers.BatchNormalization()
        self.dropout2 = K.layers.Dropout(rate=0.4)
        self.conv3 = K.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', activation='tanh')

    def call(self, inputs, training=True):
        """
        Forward pass of the generator
        - latent_inputs: Random noise from the latent space, using for generating images
        - label_inputs: Labels for the images to be generated
        - training: Boolean flag for whether training or testing
        """
        latent_inputs, label_inputs = inputs

        # Latent Inputs Layer (Dense Layer + BatchNorm + ReLU + Reshape)
        x1 = self.dense1(latent_inputs)
        x1 = self.bn1(x1, training=training)
        x1 = K.layers.LeakyReLU()(x1)
        x1 = self.reshape1(x1)

        # Process label inputs
        x2 = self.embedding(label_inputs)
        x2 = self.dense2(x2)
        x2 = self.bn2(x2, training=training)
        x2 = K.layers.LeakyReLU()(x2)
        x2 = self.reshape2(x2)

        #
        merged_inputs = K.layers.Concatenate()([x1, x2])
        x = self.conv1(merged_inputs)
        x = self.bn3(x, training=training)
        x = K.layers.LeakyReLU()(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.bn4(x, training=training)
        x = K.layers.LeakyReLU()(x)
        x = self.dropout2(x, training=training)
        x = self.conv3(x)

        return x

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({"latent_dim": self.latent_dim, "n_classes": self.n_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

#=====================================================================================================================


@K.saving.register_keras_serializable()
class Discriminator(K.Model):
    """
    Discriminator component of AC-GAN for MNIST dataset

    Args:
    - n_classes: Number of classes(labels) in the dataset (default=10) which predicted (discriminated) by the Discriminator
    """

    def __init__(self, n_classes=10, name="discriminator", **kwargs):
        super(Discriminator, self).__init__(name=name, **kwargs)
        self.n_classes = n_classes

        # Define layers
        self.gaussian_noise = K.layers.GaussianNoise(stddev=0.2)
        self.conv1 = K.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.bn1 = K.layers.BatchNormalization()
        self.dropout1 = K.layers.Dropout(rate=0.4)
        self.conv2 = K.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same', use_bias=False)
        self.bn2 = K.layers.BatchNormalization()
        self.dropout2 = K.layers.Dropout(rate=0.4)

        # flatten layer
        self.flatten = K.layers.Flatten()

        # Output layers: 2 Dense Layer for validity and label prediction
        self.dense1 = K.layers.Dense(units=1, activation='sigmoid') # dense layer for validity the image
        self.dense2 = K.layers.Dense(units=n_classes, activation='softmax') # dense layer for classifying the label

    # @tf.function
    def call(self, inputs, training=True):
        """
        Forward pass of the discriminator
        Args:
        - inputs: Input images to be discriminated. Passing the input (generated by the Generator) through the Discriminator
        and output the validity and label prediction
        - training: Boolean flag for whether training or testing

        Returns:
        - validity: Validity of the input image that the discriminator predicts
        - label: Label of the input image that the discriminator predicts
        """
        x = self.gaussian_noise(inputs)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = K.layers.LeakyReLU()(x)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = K.layers.LeakyReLU()(x)
        x = self.dropout2(x, training=training)

        x = self.flatten(x)

        # Output layers
        validity = self.dense1(x)
        label = self.dense2(x)

        return validity, label

    def get_config(self):
        config = super(Discriminator, self).get_config()
        config.update({"n_classes": self.n_classes})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
#===================================================================================================================


#===================================================================================================================
#                                           AC-GAN
#===================================================================================================================
@K.saving.register_keras_serializable()
class ACGAN(K.Model):
    def __init__(self, generator, discriminator, latent_dim, n_classes=10, name="acgan", **kwargs):
        super(ACGAN, self).__init__(name=name, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.generator_optimizer = K.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
        self.discriminator_optimizer = K.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

        # Define loss functions with label smoothing
        self.binary_loss = K.losses.BinaryCrossentropy(label_smoothing=0.25) #
        self.sparse_categorical_loss = K.losses.SparseCategoricalCrossentropy()

    def compile(self):
        super(ACGAN, self).compile()

        # Set the discriminator to not trainable initially
        self.discriminator.trainable = False

    def call(self, inputs, training=False):
        """
        Forward pass of the ACGAN model.

        Args:
        - inputs: A list containing [latent_inputs, label_inputs] which refers to the random noise
        - training: Boolean flag for whether training or testing

        Returns:
        - discriminated_validity: Validity of the input image that the discriminator predicts
        - discriminated_label: Label of the input image that the discriminator predicts
        """
        latent_inputs, label_inputs = inputs
        generated_images = self.generator([latent_inputs, label_inputs], training=training)
        discriminated_validity, discriminated_label = self.discriminator(generated_images, training=training)
        return discriminated_validity, discriminated_label

    def train_step(self, data):
        x_batch, y_batch = data
        batch_size = tf.shape(x_batch)[0]

        # Ground Truth labels
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        mixed_labels = tf.concat([real_labels, fake_labels], axis=0)

        # Generate random noise and labels
        random_latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        random_labels = np.random.randint(0, self.n_classes, size=[batch_size])

        # Generate images
        generated_images = self.generator([random_latent_noise, random_labels], training=True)

        # ------------------------------- Train the Discriminator ---------------------------------------------
        self.discriminator.trainable = True
        with tf.GradientTape() as tape:
            # Get discriminator outputs for real images
            real_validity, real_label = self.discriminator(x_batch, training=True)
            # Get discriminator outputs for generated images
            fake_validity, fake_label = self.discriminator(generated_images, training=True)

            # Concatenate real and fake outputs
            discriminated_validity = tf.concat([real_validity, fake_validity], axis=0)
            discriminated_label = tf.concat([real_label, fake_label], axis=0)
            mixed_generated_labels = tf.concat([y_batch, random_labels], axis=0)

            discriminator_loss = [
                self.binary_loss(mixed_labels, discriminated_validity), #
                self.sparse_categorical_loss(mixed_generated_labels, discriminated_label)
            ]
            total_discriminator_loss = tf.reduce_mean(discriminator_loss[0]) + tf.reduce_mean(discriminator_loss[1])

        gradients_D = tape.gradient(total_discriminator_loss, self.discriminator.trainable_variables)
        if None in gradients_D:
            raise ValueError("No gradients provided for some variables.")
        self.discriminator_optimizer.apply_gradients(zip(gradients_D, self.discriminator.trainable_variables))

        # ------------------------------------- Train the Generator ---------------------------------------------
        self.discriminator.trainable = False #disable the discriminator 
        with tf.GradientTape() as tape:
            # Generate images again to ensure they are within the tape context
            generated_images = self.generator([random_latent_noise, random_labels], training=True)
            discriminated_validity, discriminated_label = self.discriminator(generated_images, training=True)

            generator_loss = [
                self.binary_loss(real_labels, discriminated_validity), # label loss
                self.sparse_categorical_loss(random_labels, discriminated_label) #
            ]
            total_generator_loss = tf.reduce_mean(generator_loss[0]) + tf.reduce_mean(generator_loss[1])

        gradients_G = tape.gradient(total_generator_loss, self.generator.trainable_variables)
        if None in gradients_G:
            raise ValueError("No gradients provided for some variables.")
        self.generator_optimizer.apply_gradients(zip(gradients_G, self.generator.trainable_variables))

        return {
            "d_loss": total_discriminator_loss,
            "g_loss": total_generator_loss
        }

    def get_config(self):
        config = super(ACGAN, self).get_config()
        config.update({"latent_dim": self.latent_dim, "n_classes": self.n_classes})
        return config

    @classmethod
    def from_config(cls, config):
        generator_config = config.pop('generator')
        discriminator_config = config.pop('discriminator')
        # Assuming generator and discriminator are classes that can be instantiated directly
        generator = Generator.from_config(generator_config)  # Replace with actual instantiation if different
        discriminator = Discriminator.from_config(discriminator_config)  # Replace with actual instantiation if different

        return cls(generator=generator, discriminator=discriminator, **config)

    def generate_images(self, latent_space, labels):
        """
        Generate images from the latent space and labels. Using Generator only.
        Args:
        - latent_space: Random noise from the latent space
        - labels: Labels for the images to be generated
        """
        return self.generator([latent_space, labels], training=False)

    def discriminate_images(self, images):
        """
        Discriminate the images using the Discriminator.
        Args:
        - images: Images to be discriminated
        """
        return self.discriminator(images, training=False)
    

#===================================================================================================================
def evaluate_acgan(acgan, x_test, y_test, batch_size=32):
    """Evaluates the AC-GAN using the discriminator's auxiliary classifier."""
    _, aux_output = acgan.discriminator.predict(x_test, batch_size=batch_size)
    predicted_labels = np.argmax(aux_output, axis=1)
    accuracy = np.mean(predicted_labels == y_test)
    return accuracy


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

    #--------------------------------------------------------------------------------------------------------------
    #                                               TRAIN AC-GAN
    #--------------------------------------------------------------------------------------------------------------

    # Generator
    generator = Generator(latent_dim=128, n_classes=10)

    # Discriminator
    discriminator = Discriminator(n_classes=10)

    # # # Dummy pass to ensure variables exist
    _ = generator([tf.zeros((1, latent_dim)), tf.zeros((1,), dtype=tf.int32)], training=False)
    _ = discriminator(tf.zeros((1, 28, 28, 1)), training=False)

    acgan = ACGAN(generator, discriminator, latent_dim=128, n_classes=10)
    acgan.compile()


    # ---------------- Train process -----------------------------------------------------------------------

    batches_per_epoch = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        for i, (x_batch, y_batch) in enumerate(train_dataset):
            losses = acgan.train_step([x_batch, y_batch])

            if i % 100 == 0:
                print(f"Batch {i}/{batches_per_epoch}, Discriminator Loss: {losses['d_loss']}, Generator Loss: {losses['g_loss']}")

        print(f"\nEpoch ({epoch+1}/{epochs}): \n Discriminator Loss: {losses['d_loss']}, Generator Loss: {losses['g_loss']}\n")

    print("Training complete!")

    # # -------------------------- Save the model ---------------------------------------------------------
    print("Saving generator and discriminator")
    discriminator.trainable = True
    try:
        generator.save("../models/acgan/generator.keras")

        discriminator.save("../models/acgan/discriminator.keras")
    except Exception as e:
        print(f"Error saving the generator, discrimator: {e}")

    # Save the AC-GAN model
    print("Saving the model")
    discriminator.trainable = False
    try:
        acgan.save("../models/acgan/acgan.keras")
    except Exception as e:
        print(f"Error saving the generator, discrimator: {e}")
    # #----------------------------------------------------------------------------

    # # Evaluate the AC-GAN
    accuracy = evaluate_acgan(acgan, x_test, y_test)
    print(f"AC-GAN Test Accuracy: {accuracy * 100:.2f}%")

    # Generate and save some images from the AC-GAN ===========================================
    digits_per_class = 10
    random_noise = tf.random.normal(shape=[digits_per_class * n_classes, latent_dim])
    digit_targets = np.array([target for target in range(n_classes) for _ in range(digits_per_class)])
    generated_digits = generator.predict([random_noise, digit_targets])
    generated_digits = ((generated_digits + 1) * 127.5).astype(np.uint8)

    save_dir = "generated_acgan"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, digit in enumerate(generated_digits):
        img = K.preprocessing.image.array_to_img(digit.reshape(28, 28, 1))
        img.save(os.path.join(save_dir, f"digit_{i}.png"))

    #==================================================================================================================