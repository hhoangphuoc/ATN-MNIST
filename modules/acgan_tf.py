import tensorflow as tf
import keras as K
import tensorflow_datasets as tfds


import numpy as np
import matplotlib.pyplot as plt

import os


# SAVE AND LOAD FUNCTIONS
# Save and Load the model

def save(gan, generator, discriminator, model_folder, prefix="ACGAN"):
        """
        Save the model weights
        Args:
        - path: Path to save the model weights
        - prefix: Prefix for the model weights
        """
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        
        K.models.save_model(generator, f"{model_folder}/{prefix}/generator")
        K.models.save_model(discriminator, f"{model_folder}/{prefix}/discriminator")

        # save model
        K.models.save_model(gan, f"{model_folder}/{prefix}/model")

def load(model_folder, prefix="ACGAN"):
    """
    Load the model weights
    Args:
    """
    generator = K.models.load_model(f"{model_folder}/{prefix}/generator")
    discriminator = K.models.load_model(f"{model_folder}/{prefix}/discriminator")
    
    gan = K.models.load_model(f"{model_folder}/{prefix}/model")

    gan.summary()
    generator.summary()
    discriminator.summary()

    return generator, discriminator, gan

#================================================================================================

# ACGAN MODEL

class Generator(K.Model):
    """
    Generator component of AC-GAN for MNIST dataset

    Args:
    - latent_dim: Dimension of the latent space (generated as noise)
    - n_classes: Number of classes(labels) in the dataset (default=10)

    inherited from https://github.com/kochlisGit/Generative-Adversarial-Networks/blob/main/mnist-digits-acgan/digits-acgan.py

    """
    def __init__(self, latent_dim, n_classes=10):
        super(Generator, self).__init__()
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

    @tf.function
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

class Discriminator(K.Model):
    """
    Discriminator component of AC-GAN for MNIST dataset

    Args:
    - n_classes: Number of classes(labels) in the dataset (default=10) which predicted (discriminated) by the Discriminator
    """
    def __init__(self, n_classes=10):
        super(Discriminator, self).__init__()
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

    @tf.function
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
    

class ACGAN(K.Model):
    def __init__(self, generator, discriminator, latent_dim, n_classes=10):
        super(ACGAN, self).__init__()
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

        # Compile the combined model
        self.compile(
            optimizer=self.generator_optimizer,
            loss=[self.binary_loss, self.sparse_categorical_loss]
        )

    def train_step(self, data):
        """
        Training step for the ACGAN model
        Args:
        - data: A batch of real images getting from the dataset (i.e. MNIST), this contains the images and labels,
        and the corresponding shape and size of the images
        """
        x_batch, y_batch = data
        batch_size = tf.shape(x_batch)[0]

        # =========================== Ground Truth labels =======================================
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        mixed_labels = tf.concat([real_labels, fake_labels], axis=0)
        #========================================================================================



        # ====================== Generate the Noise for Discriminator ===========================

        # Generate random noise and random labels from the latent space
        random_latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        random_labels = tf.random.uniform(shape=[batch_size], minval=0, maxval=self.n_classes, dtype=tf.int32) # Categorical labels

        # Generate images from random noise and labels by Generator
        generated_images = self.generator([random_latent_noise, random_labels], training=True)

        # Mixed the real and generated images and labels for Discriminator (Concatenating)
        mixed_images = tf.concat([x_batch, generated_images], axis=0) 
        mixed_generated_labels = tf.concat([y_batch, random_labels], axis=0)

        #========================================================================================


        # =========================== Train the Discriminator ====================================
        self.discriminator.trainable = True # Set the discriminator to trainable

        with tf.GradientTape() as tape:
            discriminated_validity, discriminated_label = self.discriminator(mixed_images, training=True)

            discriminator_loss = [
                self.binary_loss(mixed_labels, discriminated_validity), # validity loss
                self.sparse_categorical_loss(mixed_generated_labels, discriminated_label) #label loss
            ]

            total_discriminator_loss = tf.reduce_mean(discriminator_loss[0]) + tf.reduce_mean(discriminator_loss[1])

        gradients_D = tape.gradient(total_discriminator_loss, self.discriminator.trainable_variables)

        self.discriminator_optimizer.apply_gradients(zip(gradients_D, self.discriminator.trainable_variables))

        #========================================================================================



        # =========================== Train the Generator =================================================
        self.discriminator.trainable = False # Set the discriminator to not trainable

        with tf.GradientTape() as tape:
            generated_images = self.generator([random_latent_noise, random_labels], training=True)
            discriminated_validity, discriminated_label = self.discriminator(generated_images, training=False)

            generator_loss = [
                self.binary_loss(real_labels, discriminated_validity),
                self.sparse_categorical_loss(random_labels, discriminated_label)
            ]

            total_generator_loss = tf.reduce_mean(generator_loss[0]) + tf.reduce_mean(generator_loss[1])

        gradients_G = tape.gradient(total_generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_G, self.generator.trainable_variables))

        #========================================================================================
        
        return {
            "d_loss": total_discriminator_loss,
            "g_loss": total_generator_loss
        }
    

    def generate_images(self, latent_space, labels):
        """
        Generate images from the latent space and labels. Using Generator only.
        Args:
        - latent_space: Random noise from the latent space
        - labels: Labels for the images to be generated
        """
        return self.generator([latent_space, labels], training=False)
    



# =================================================================================================

# AT-GAN MODELS: Extended from ACGAN for Adversarial Attack
class TargetClassifier(K.Model):
    """
    Target Classifier for the AT-GAN model.
    This simply acts as the classifier for the input images (MNIST) of either real or generated images.
    Using as the target for the attack.
    """
    def __init__(self, num_classes=10):
        super(TargetClassifier, self).__init__()

        # Classifier Layers
        self.conv1 = K.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))
        self.pool1 = K.layers.MaxPooling2D((2, 2))
        self.conv2 = K.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = K.layers.MaxPooling2D((2, 2))

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class Attack_Generator(K.Model):
    """
    G_attack simply a copy of AC-GAN Generator, and used for the adversarial attack.
    Which transfering the output of the Generator to the Target Classifier.
    """
    def __init__(self, generator):
        super(Attack_Generator, self).__init__()
        self.generator = generator

    def call(self, inputs, training=False):
        return self.generator(inputs, training=training)


#================================================================================================

# ATGAN MODEL

class ATGAN:
    def __init__(self, G_original, G_attack, f_target, noise_size, lambda_adv_at=2.0, lambda_dist=1.0):
        self.G_original = G_original # Original Generator (G_original)
        self.G_attack = G_attack # Adversarial Generator (G_attack)
        self.f_target = f_target    # Target Classifier (f_target)

        self.noise_size = noise_size # latent space size

        self.lambda_adv_at = lambda_adv_at  # lambda for adversarial loss
        self.lambda_dist = lambda_dist     # lambda for distance loss

        self.optimizer_G_attack = K.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.sparse_categorical_loss = K.losses.SparseCategoricalCrossentropy()

    @tf.function
    def train_step_atgan(self, images, target_labels):
        batch_size = tf.shape(images)[0]

        with tf.GradientTape() as g_attack_tape:
            z = tf.random.normal([batch_size, self.noise_size])

            # Generate adversarial images
            adv_images = self.G_attack([z, target_labels], training=True)

            # Target classifier's prediction on adversarial images
            pred_adv = self.f_target(adv_images, training=False)

            # 1. Adversarial Loss (La) ========================================================
            
            la_loss = tf.reduce_mean(
                self.sparse_categorical_loss(target_labels, pred_adv)
            )

            # 2. Distance Loss (Ld) ========================================================
            # Add Gaussian noise
            noise = tf.random.normal(shape=tf.shape(adv_images), mean=0.0, stddev=0.1)
            adv_images_noisy = adv_images + noise

            # Original images generated by G_original
            orig_images = self.G_original([z, target_labels], training=False)

            ld_loss = tf.reduce_mean(tf.square(orig_images - adv_images_noisy))

            # Total adversarial loss for G_attack
            g_attack_loss = self.lambda_adv_at * la_loss + self.lambda_dist * ld_loss

        # Calculate G_attack gradients
        g_attack_gradients = g_attack_tape.gradient(g_attack_loss, self.G_attack.trainable_variables)
        self.optimizer_G_attack.apply_gradients(zip(g_attack_gradients, self.G_attack.trainable_variables))

        return g_attack_loss, la_loss, ld_loss

# =================================================================================================
#                                       MAIN TRAINING
# =================================================================================================

# Load the dataset
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()

# Normalize to [-1, 1] and add channel dimension
x_train = x_train.astype(np.float32) / 127.5 - 1
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.astype(np.float32) / 127.5 - 1
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Convert labels to one-hot encoding
y_train_one_hot = K.utils.to_categorical(y_train, num_classes=10)
y_test_one_hot = K.utils.to_categorical(y_test, num_classes=10)


# ================================== Training the AC-GAN ==================================
# hyperparameters
latent_dim = 128
n_classes = 10
batch_size = 64
epochs = 100

# Create  Target Classifier
f_target = TargetClassifier(num_classes=10)
f_target.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

f_target.fit(x_train, y_train_one_hot, epochs=epochs, batch_size=batch_size)


# Create the AC-GAN model
generator = Generator(latent_dim, n_classes)
discriminator = Discriminator(n_classes)

acgan = ACGAN(generator, discriminator, latent_dim, n_classes)

acgan.compile()

# Training
# # Create the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot)).shuffle(buffer_size=x_train.shape[0])
inputs = train_dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

batches_per_epoch = x_train.shape[0] // batch_size

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")

    for i, (x_batch, y_batch) in enumerate(inputs):
        losses = acgan.train_step([x_batch, y_batch])

        print(f"\rBatch {i+1}/{batches_per_epoch} \n - Discriminator Loss: {losses['d_loss']:.4f} - Generator Loss: {losses['g_loss']:.4f}", end="")


# Save the model
model_folder = "models"
save(acgan, generator, discriminator, model_folder, prefix="ACGAN")


# ================================== Training the AT-GAN ==================================
# Train AT-GAN
epochs_atgan = 50

# Create G_attack and AT-GAN
G_attack_instance = Attack_Generator(generator)
atgan = ATGAN(generator, G_attack_instance, f_target, latent_dim)

for epoch in range(epochs_atgan):
    print('\nTraining AT-GAN on epoch', epoch + 1)
    for i, (x_batch, _) in enumerate(inputs):
        target_labels = np.random.randint(0, n_classes, size=[batch_size])
        g_attack_loss, la_loss, ld_loss = atgan.train_step_atgan(x_batch, target_labels)
        
        print(f"\rBatch {i+1}/{batches_per_epoch} - G_attack Loss: {g_attack_loss:.4f}, La Loss: {la_loss:.4f}, Ld Loss: {ld_loss:.4f}", end="")