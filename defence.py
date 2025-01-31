import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
import keras as K
from time import time

from modules.target import CNNVariation, TargetClassifier
from modules.acgan import ACGAN, Generator, Discriminator
from modules.atgan import ATGAN, Attack_Generator

def create_partly_adv_data(adversarial_generator, original_model, batch_images, fraction=0.5, noise_size=128):
    batch_size = batch_images.shape[0]
    num_adversarial = int(fraction * batch_size)

    batch_images = np.array(batch_images)
    original_batch_images = batch_images.copy()

    # Generate adversarial examples
    preds = original_model.predict(batch_images, verbose=0)
    target_labels = np.argmax(preds, axis=1)  # Use predicted labels as targets


    # Select random indices within the batch
    indices = np.random.choice(batch_size, num_adversarial, replace=False)

    # Generate random noise
    z = tf.random.normal([batch_size, noise_size])
    adversarial_examples = adversarial_generator([z, target_labels])

    # Create a new batch with a mix of original and adversarial examples
    batch_images[indices] = adversarial_examples[:num_adversarial]
    # batch_images[indices] = adversarial_examples

    # Convert back to TensorFlow tensor
    batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
    
    return batch_images

def cnn_train_step(generator, original_model, batch_images, batch_labels):
    mixed_images = create_partly_adv_data(
        adversarial_generator=generator, 
        original_model=original_model, 
        batch_images=batch_images, 
        fraction=0.5)
    loss, accuracy = original_model.train_on_batch(mixed_images, batch_labels)
    return loss, accuracy  # Return both loss and accuracy


#=================================================================================================================
if __name__ == "__main__":
    # =================================================================================================================
    #                             Load and Preprocess Dataset (SIMILAR TO ALL OTHER MODELS)
    # =================================================================================================================

    latent_dim = 128 #noise size
    num_epochs = 100  # Adjust as needed
    batch_size = 32  # Adjust as needed

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

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=x_train.shape[0])
    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)


    # Load the CNN model
    # defence_cnn = DefenceCNN()
    # model = defence_cnn.build_model()
    # model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model.fit(train_dataset, epochs=20, )  # Train for one epoch to initialize the model

    model = K.models.load_model("./models/target/cnn_variation.keras") #TODO: THIS ALREADY PRE_TRAINED WITH 20 EPOCHS - DO WE NEED TO RETRAIN?
    # Load the ATNs
    generator = K.models.load_model("./models/acgan/generator.keras")
    #================================================================================================================

    # Train the model
    for epoch in range(num_epochs):
        cnn_start_time = time()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_data_iterator = iter(train_dataset)  # Create an iterator
        total_loss = 0
        total_accuracy = 0
        num_batches = len(train_dataset)  # Total number of batches

        for batch_number in range(num_batches):  # Loop over batches
            try:
                # Get the next batch
                x, labels = next(train_data_iterator)
                loss, accuracy = cnn_train_step(
                    generator=generator,
                    original_model=model,
                    batch_images=x, 
                    batch_labels=labels)  

                # Accumulate metrics
                total_loss += loss
                total_accuracy += accuracy

                # Progress bar display
                progress = int((batch_number + 1) / num_batches * 30)  # 30-character progress bar
                sys.stdout.write(
                    f"\r[{'=' * progress}{'.' * (30 - progress)}] "
                    f"{batch_number + 1}/{num_batches} - loss: {loss:.4f} - accuracy: {accuracy:.4f}"
                )
                sys.stdout.flush()

            except StopIteration:
                break  # In case the iterator runs out of data

        # End of epoch metrics
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        elapsed_time = time() - cnn_start_time
        print(f"\nEpoch {epoch + 1} completed in {elapsed_time:.2f}s - loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f}")

    # Save the model
    model.save("./models/target/cnn_variation_defence.keras")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_accuracy:.4f}")
    
    print("Training completed!")