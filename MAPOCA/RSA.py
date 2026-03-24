#=== Implémentation du modèle Residual self-Attention (RSA) ===#

import torch
import torch.nn as nn
import torch.nn.functional as F

class RSAModule(nn.Module):
    def __init__ (self, feature_dim=512, num_heads= 4):
        super().__init__()
        self.feature_dim = feature_dim

        # Multi-Head Attention : permet de regarder différentes zones du jeu en même temps
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

        # Layer Normalization pour stabiliser l'entraînement multi-agent
        self.norm = nn.LayerNorm(feature_dim)

        # feedforward pour traiter les features après l'attention

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self,x):
        """
        x: tenseur de forme (batch_size, num_agents, feature_dim)
        """

        # self-attention: utilisation du principe de "query", "key" et "value" pour permettre à chaque agent de se concentrer sur les autres agents
        attn_output, _ = self.attention(x,x,x)

        # connexion résiduelle et normalisation

        x = self.norm(x + attn_output)

        # 2ème bloc résiduel avec Feedforward

        ff_output = self.fc(x)
        out = self.norm(x + ff_output)

        return out
