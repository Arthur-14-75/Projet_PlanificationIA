# Projet de planification en IA

## Points importants à suivre pour charger le jeu Ice Hockey en mode multi agents

L'environnement gymnasium est normalement celui utilisé pour charger les jeux ATARI, mais pour nous, on va devoir utiliser PettingZoo car on fait du multi agent.

Etapes à suivre:
- Installer PettingZoo dans le terminal de VS code: pip install "pettingzoo[atari,accept-rom-license]"
- Si vous êtes sur Mac, il faut installer cmake directement sur votre machine avec brew: brew install cmake
- Ensuite, il faut installer le fichier du jeu: pip install "AutoROM[accept-rom-license]" puis  
AutoROM --accept-license