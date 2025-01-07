import torch
import torch.nn as nn

# Basic CNN (BCNN)
class BasicCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.5):
        super(BasicCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # Reshape if the input x has the shape of [batch_size, 784]
        # reshape the input x to [batch_size, 1, 28, 28]
        torch.reshape(x, (-1, 1, 28, 28))
        x = self.conv_layers(x)
        torch.reshape(x, (-1, 7 * 7 * 64))
        x = self.fc_layers(x)
        return nn.functional.softmax(x, dim=1)