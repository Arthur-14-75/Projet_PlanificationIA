try:
    from MAPOCA.observation_encoding import Impala_CNN
except ImportError:
    from observation_encoding import Impala_CNN

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAgentActors(nn.Module):
    def __init__ (self, observation_shape=(4, 84, 84), feature_dim=512, action_dim=18):
        super(MultiAgentActors, self).__init__()

        # encodeur: on partage l'instance avec le critic ou on en créé une nouvelle
        self.encoder = Impala_CNN(input_shape=observation_shape, feature_dim=feature_dim)

        # Tête de politique (policy head)

        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, observations):
        """
        obs: Tenseur (batch_size, 4, 84, 84)
        Retourne : Logits des actions
        """
        features = self.encoder(observations)
        logits = self.policy_head(features)
        return logits
    
    def get_action(self, observations, deterministic = False):
        """
        Choisit une action à partir d'une observation
        """
        logits = self.forward(observations)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            return torch.argmax(probs, dim=1)
        else:
            # Échantillonnage selon la distribution (pour l'exploration)
            dist = torch.distributions.Categorical(probs)
            return dist.sample()


