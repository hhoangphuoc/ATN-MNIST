import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from modules.ae import BasicAE
from modules.cnn import BasicCNN



# Adversarial Transformation Network (ATN) with AAE
class ATN(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, dropout_prob=0.5, loss_beta=0.1, learning_rate=0.01):
        super(ATN, self).__init__()
        self.loss_beta = loss_beta
        self.learning_rate = learning_rate

        self.autoencoder = BasicAE(input_shape)
        self.target_adv = BasicCNN(num_classes, dropout_prob)
        self.target = BasicCNN(num_classes, dropout_prob)
        self.target.load_state_dict(self.target_adv.state_dict())  # Initialize target with the same parameters

        # Optimizers (using Adam)
        self.optimizer_atn = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        self.optimizer_adv = optim.Adam(self.target_adv.parameters(), lr=1e-4) # Optimizer for target_adv following the basic cnn
        self.optimizer_target = optim.Adam(self.target.parameters(), lr=1e-4) # Optimizer for target
    def forward(self, x):
        return self.autoencoder(x)

    def train_atn(self, data, rerank):
        # ATN Optimization
        self.optimizer_atn.zero_grad()

        y_pred = self.autoencoder(data)
        # y_true = data.view(-1, 784)  # Reshape to [batch_size, 784]
        y_true = torch.reshape(data, (-1, 784))

        Lx = self.loss_beta * torch.sum(torch.sqrt(torch.sum((y_pred - y_true)**2, dim=1)))
        target_adv_pred = self.target_adv(y_pred)

        Ly = torch.sum(torch.sqrt(torch.sum((target_adv_pred - rerank)**2, dim=1)))
        atn_loss = Lx + Ly

        atn_loss.backward()
        self.optimizer_atn.step()
        
        return atn_loss.item()

    def train_adv(self, data, labels_gt):
        self.optimizer_adv.zero_grad()

        transformed_data = self.autoencoder(data).detach() # Detach to avoid affecting ATN gradients

        adv_pred = self.target_adv(transformed_data)
        logprob = torch.log(adv_pred + 1e-12)
        adv_loss = -torch.sum(labels_gt * logprob)

        adv_loss.backward()
        self.optimizer_adv.step()
        return adv_loss.item()

    def train_target(self, data, labels_gt):
      self.optimizer_target.zero_grad()
      target_pred = self.target(data)
      logprob = torch.log(target_pred + 1e-12)
      target_loss = -torch.sum(labels_gt * logprob)
      target_loss.backward()
      self.optimizer_target.step()
      return target_loss.item()

    def get_accuracy(self, data, labels_gt):
        with torch.no_grad():
            predictions = self.target(data)
            correct_predictions = torch.equal(torch.argmax(labels_gt, dim=1), torch.argmax(predictions, dim=1))
            accuracy = torch.mean(correct_predictions.float())
        return accuracy.item()

    def save(self, path, prefix="ATN_"):
        torch.save(self.autoencoder.state_dict(), f"{path}/{prefix}basic_ae.pth")
        torch.save(self.target.state_dict(), f"{path}/{prefix}basic_cnn.pth")
        torch.save(self.target_adv.state_dict(), f"{path}/{prefix}basic_cnn_adv.pth")

    def load(self, path, prefix="ATN_"):
        self.autoencoder.load_state_dict(torch.load(f"{path}/{prefix}basic_ae.pth"))
        self.target.load_state_dict(torch.load(f"{path}/{prefix}basic_cnn.pth"))
        self.target_adv.load_state_dict(torch.load(f"{path}/{prefix}basic_cnn_adv.pth"))