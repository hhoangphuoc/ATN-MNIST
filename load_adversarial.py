import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import keras as K
import numpy as np
import matplotlib.pyplot as plt
from modules.target import TargetClassifier
from modules.acgan import ACGAN, Generator, Discriminator
from modules.atgan import ATGAN, Attack_Generator
# ==================================================================================================================
#                            HELPER FUNCTIONS
#===================================================================================================================
def display_images(
    original_images, #x_batch
    adversarial_images, #adversarial_x_batch
    y_true,
    y_pred_original,
    y_pred_adversarial,
    y_pred_original_label,  # Simplified target label
    save_path="adversarial_images",
):  
    print("Total images:", len(original_images))
    num_images = min(10, len(original_images))
    fig = plt.figure(figsize=(15, 5)) 
    for i in range(num_images):

        # Original image
        plt.subplot(2, num_images, i + 1)  # 2 rows, num_images columns, i-th subplot
        plt.imshow(tf.squeeze(original_images[i]), cmap="gray")
        plt.title(f"Label: {y_true[i].numpy()}, Pred: {y_pred_original[i].numpy()}")
        plt.axis("off")

        # Adversarial image
        plt.subplot(2, num_images, i + 1 + num_images)  # 2 rows, num_images columns, i-th subplot
        plt.imshow(tf.squeeze(adversarial_images[i]), cmap="gray")
        plt.title(f"\nTarget: {y_pred_original_label[i]}, Pred: {y_pred_adversarial[i].numpy()}")
        plt.axis("off")

        # Save the figure
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f"adversarial_image.png"))

    # Close the figure to free up memory
    # plt.show()
    plt.close(fig)
#================================================================================================================

def evaluate_atn(
    # atgan,
    adversarial_generator, # adversarial model
    original_model,
    dataset,
    noise_size=128,
    n_classes=10,
    num_batches=2,
    adversarial_type = "gan",
    image_save_path="adversarial_images",
):
    target_classifier_fooled = 0
    total_samples = 0
    i = 0
    for x_batch, y_batch in dataset.take(num_batches):
        batch_size = x_batch.shape[0]
        z = tf.random.normal([batch_size, noise_size])

        # 1. Generate target labels (using original model's predictions)
        preds = original_model.predict(x_batch, verbose=0)
        target_labels_batch = np.argmax(preds, axis=1)  # Use predicted labels as targets

        # 2. Generate adversarial examples ==================================================
        
        # AC-GAN Attack
        if adversarial_type == "gan":
            adversarial_x_batch = adversarial_generator([z, target_labels_batch], training=False)

        # 3. Get predictions
        y_pred_original = tf.argmax(
            original_model.predict(x_batch, verbose=0), axis=1
        )
        y_pred_adversarial = tf.argmax(
            original_model.predict(adversarial_x_batch, verbose=0), axis=1
        )

        # 4. True labels (cast to int64)
        y_true = tf.cast(y_batch, tf.int64)  # Cast y_batch to tf.int64

        # 5. Count successful attacks for fooling rate
        target_classifier_fooled += np.sum(
            y_pred_adversarial.numpy() == target_labels_batch
        )
        total_samples += batch_size

        # 6. Print results for comparison
        print("---------------------------------------------------------")
        print(f"Batch {i + 1}/{num_batches}")
        print("True Labels:", y_true.numpy())
        print("Original Predictions:", y_pred_original.numpy())
        print("Adversarial Predictions:", y_pred_adversarial.numpy())
        print("---------------------------------------------------------")
        
        i += 1

        # 7. Find interesting cases (original correct, adversarial incorrect)
        correct_original = tf.equal(y_pred_original, y_true)
        incorrect_adversarial = tf.not_equal(y_pred_adversarial, y_true)
        interesting_indices = tf.where(
            tf.logical_and(correct_original, incorrect_adversarial)
        )
        interesting_indices = tf.squeeze(interesting_indices, axis=1)

        # 8. Filter data for visualization
        filtered_x_batch = tf.gather(x_batch, interesting_indices)
        filtered_adversarial_x_batch = tf.gather(
            adversarial_x_batch, interesting_indices
        )
        filtered_y_true = tf.gather(y_true, interesting_indices)
        filtered_y_pred_original = tf.gather(y_pred_original, interesting_indices)
        filtered_y_pred_adversarial = tf.gather(
            y_pred_adversarial, interesting_indices
        )

        # 9. Display images
        if len(filtered_x_batch) > 0:  # Only plot if there are interesting cases
            display_images(
                filtered_x_batch,
                filtered_adversarial_x_batch,
                filtered_y_true,
                filtered_y_pred_original,
                filtered_y_pred_adversarial,
                filtered_y_pred_original.numpy(),  # Pass original predictions as target labels
                save_path=image_save_path,
            )
        else:
            print("No interesting cases (where original is correct and adversarial is wrong) in this batch.")

    # Calculate the success rate
    fooling_rate = (target_classifier_fooled / total_samples) * 100
    print(f"AT-GAN Attack Success Rate (Fooling Rate): {fooling_rate:.2f}%")
    return fooling_rate

#==============================================================================================================


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
    #================================================================================================================

    #-----------------------------------------------------------------------------------------------------------------
    #                                           LOAD RELATED MODELS
    #-----------------------------------------------------------------------------------------------------------------
    
    adversarial_generator = K.models.load_model("./models/acgan/generator.keras")
    target_classifier = K.models.load_model("./models/target/f_target.keras")

    #-----------------------------------------------------------------------------------------------------------------

    # Evaluate the atn with plot
    try:
        attack_success_rate = evaluate_atn(
            adversarial_generator=adversarial_generator,
            original_model=target_classifier,
            dataset=test_dataset,
            num_batches=32,
            image_save_path="./data/adversarial_images",
        )
        print(f"AT-GAN Attack Success Rate: {attack_success_rate:.2f}%")
    except Exception as e:
        print(f"Error evaluating AT-GAN with plotting: {e}")