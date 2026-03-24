# ==== Observation Encoding ====

# Ce programme permet des traiter les images du jeu, consiste à implémenter un CNN
#Pour l'architecure du CNN, on ne pourra pas utiliser des modèles trop lourds comme ResNet ou VGG, car ils sont conçus pour des tâches de classification d'images à grande échelle et peuvent être trop complexes pour notre tâche spécifique de traitement d'images de jeu.
# Nous allons donc utliser une architecture plus légère, mais tout en prenant garde au risque de disparition de gradients
# Nous allons donc utiliser l'architecture Impala CNN, compromis entre le Nature CNN et le ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual_block(nn.Module):

    def __init__ (self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out+x
    
class Impala_Layer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = Residual_block(out_channels)
        self.res2 = Residual_block(out_channels)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.maxpool(out)
        out = self.res1(out)
        out = self.res2(out)
        return out

class Impala_CNN(nn.Module):

    def __init__(self, input_shape=(4,84,84), feature_dim=512):
        super().__init__()

        self.layer1 = Impala_Layer(input_shape[0], 16)
        self.layer2 = Impala_Layer(16, 32)
        self.layer3 = Impala_Layer(32, 32)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, feature_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = x/255.0 # Normalisation des pixels entre 0 et 1
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(x)