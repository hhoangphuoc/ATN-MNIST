import torch
import torch.nn as nn
import torch.optim as optim

# Basic Autoencoder (BAE)
class BasicAE(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(BasicAE, self).__init__()
        self.input_shape = input_shape

        # Define encoder layers
        # Encoder Layers: 8 Convolutional Layers with ReLU activation
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Decoder Layers: 1 Convolutional Layer with Tanh activation
        self.decoder = nn.Sequential(
            nn.Conv2d(48, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return torch.reshape(decoded, (-1, 748))
