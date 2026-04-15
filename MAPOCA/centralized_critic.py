#==== Le but de ce programme est de prendre les observations de tous les agentset de sortir une valeur Q ou V pour chaque agent, tout en tenant compte des actions des autres.
from MAPOCA.observation_encoding import Impala_CNN
from MAPOCA.RSA import RSAModule

import torch
import torch.nn as nn
import torch.nn.functional as F

class Centralized_critic(nn.Module):

    def __init__ (self, observation_shape =(4, 84,84), feature_dim = 512, num_heads = 4, num_agents = 2):
        super(Centralized_critic, self).__init__()

        self.num_agents = num_agents
        self.feature_dim = feature_dim

        # encodeur partagé

        self.encoder = Impala_CNN(input_shape=observation_shape, feature_dim= feature_dim)

        # module de communication/attention

        self.rsa = RSAModule(feature_dim=feature_dim, num_heads=num_heads)

        # tête de la valeur: prend la sortie du RSA pour chaque agent et prédit sa valeur

        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256,1) # valeur V(s) pour l'agent i
        )

    
    def forward(self, observations_list):
        "observations_list: Liste de tenseurs d'observations [obs_agent_1, obs_agent_2]"
        "Chaque obs_agent a la forme (batch_size, 4, 84, 84)"

        assert len(observations_list) == self.num_agents, f"Attendu {self.num_agents} agents, reçu {len(observations_list)}"

        # encodage de chaque agent

        encoded_features = [self.encoder(observation) for observation in observations_list]

        # concaténation pour le RSA (batch, num_agents, feature_dim)

        combined_features = torch.stack(encoded_features, dim=1)

        # Interaction avec le RSA: c'est ici que le critique est centralisé

        shared_representation = self.rsa(combined_features)

        # calcul de la valeur pour chaque agent: on applique la tête de valeur sur chaque agent séparément

        values = self.value_head(shared_representation)

        return values.squeeze(-1) # Shape: (batch, num_agents)
