import tensorflow as tf
import keras as K

# Basic CNN (BCNN) model
class BasicCNN(K.Model):
    def __init__(self, num_classes=10, dropout_prob=0.5):
        super(BasicCNN, self).__init__()

        self.conv_layers = K.layers.Sequential(
            K.layers.Conv2D(1, 32, kernel_size=5, stride=1, padding=2),
            K.layers.ReLU(),
            K.layers.MaxPooling2D(kernel_size=2, stride=2),
            K.layers.Conv2D(32, 64, kernel_size=5, stride=1, padding=2),
            K.layers.ReLU(),
            K.layers.MaxPooling2D(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc_layers = K.Sequential(
            K.layers.Flatten(),
            K.layers.Dense(1024),
            K.layers.ReLU(),
            K.layers.Dropout(dropout_prob),
            K.layers.Dense(num_classes)
        )

    def forward(self, x):
        # Reshape if the input x has the shape of [batch_size, 784]
        # reshape the input x to [batch_size, 1, 28, 28]
        tf.reshape(x, (-1, 1, 28, 28))
        x = self.conv_layers(x)
        tf.reshape(x, (-1, 7 * 7 * 64))
        x = self.fc_layers(x)
        return K.layers.functional.softmax(x, dim=1)