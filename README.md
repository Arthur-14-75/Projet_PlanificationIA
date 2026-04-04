# Projet de planification en IA

## Points importants à suivre pour charger le jeu Ice Hockey en mode multi agents

L'environnement gymnasium est normalement celui utilisé pour charger les jeux ATARI, mais pour nous, on va devoir utiliser PettingZoo car on fait du multi agent.

Etapes à suivre:
- Installer PettingZoo dans le terminal de VS code: pip install "pettingzoo[atari,accept-rom-license]"
- Si vous êtes sur Mac, il faut installer cmake directement sur votre machine avec brew: brew install cmake
- Ensuite, il faut installer le fichier du jeu: pip install "AutoROM[accept-rom-license]" puis  
AutoROM --accept-license


## Organisation de MA-POCA

Rappel, l'organisation du programme s'inspire directement du pseudo-code de l'article "Intelligent Close Air Combat Design based on MA-POCA Algorithm"

1. Encodeur d'observation $g_{i}$ pour traiter des images d'ATARI. On n'utilisera pas de réseau sophistiqué comme ResNet ou ImageNet, plutôt adaptés pour des images de grande taille avec des textures complexes. En plus les modèles lourds comme ceux que j'ai cités peuvent grandement ralentir l'entrainement de MA-POCA, qui est déjà très lourd en lui-même. On va pour ça utiliser un Impala Net: c'est simplement composé de quelques couches Conv-ReLU avec un bloc résiduel pour éviiter la disparition du gradient.
2. Critique Centralisé, mécanisme RSA: 