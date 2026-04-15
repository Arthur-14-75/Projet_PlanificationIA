# Projet de planification en IA — Ice Hockey multi-agent

Implémentation de deux algorithmes d'apprentissage par renforcement multi-agent appliqués au jeu Atari Ice Hockey via PettingZoo :

- **MA-POCA** (Multi-Agent POsthumous Credit Assignment) — critique centralisé avec mécanisme RSA et encodeur Impala CNN
- **VDN** (Value Decomposition Networks) — décomposition de la valeur conjointe en valeurs individuelles par agent

---

## Installation

### Prérequis

- Python 3.10+
- Sur **Mac** : installer CMake avec Homebrew avant tout le reste : `brew install cmake`

### Dépendances

```bash
pip install -r requirements.txt
```

Cela installe : `torch`, `numpy`, `matplotlib`, `pillow`, `tqdm`, `pettingzoo[atari]`, `AutoROM`, `supersuit`.

Puis accepter la licence des ROMs Atari :

```bash
AutoROM --accept-license
```

---

## Structure du projet

```
Projet_PlanificationIA/
├── MAPOCA.ipynb                  # Notebook principal MA-POCA (entraînement + évaluation)
├── VDN_lancement.ipynb       # Launcher Colab pour VDN
├── MAPOCA/
│   ├── actor.py                  # Politique décentralisée (MultiAgentActors)
│   ├── centralized_critic.py     # Critique centralisé
│   ├── multi_agent_buffer.py     # Buffer de replay multi-agent
│   ├── observation_encoding.py   # Encodeur Impala CNN
│   └── RSA.py                    # Module RSA (Relational Self-Attention)
├── VDN/
│   ├── vdn.py                    # Baseline VDN — CLI complet, récompense brute
│   └── vdn_amélioré.py           # VDN étendu — AMP, reward shaping, multi-env
└── requirements.txt
```

---

## Lancer MA-POCA

MA-POCA s'exécute entièrement depuis le notebook **`MAPOCA.ipynb`**.

### En local (Jupyter)

```bash
jupyter notebook MAPOCA.ipynb
```

Exécuter les cellules dans l'ordre. Le notebook :
1. Installe les dépendances
2. Charge l'environnement Ice Hockey (PettingZoo)
3. Instancie les modules du package `MAPOCA/`
4. Lance la boucle d'entraînement
5. Sauvegarde les checkpoints dans `checkpoints_hockey/`

Pour **reprendre un entraînement**, modifier la cellule de chargement avec le chemin vers le checkpoint `.pth` souhaité avant d'exécuter.

### Sur Google Colab

Uploader le dossier sur Google Drive, puis ouvrir `MAPOCA.ipynb` directement dans Colab. Les chemins de checkpoints sont configurables dans les cellules dédiées.

---

## Lancer VDN

Deux variantes sont disponibles selon le contexte d'entraînement.

### `vdn.py` — Baseline (CLI complet)

Version de référence, entièrement configurable en ligne de commande.

```bash
cd VDN
python vdn.py
```

Options disponibles :

```
--episodes          Nombre d'épisodes (défaut : 500)
--max-steps         Pas max par épisode (défaut : 1000)
--lr                Learning rate (défaut : 1e-4)
--seed              Graine aléatoire (défaut : 42)
--device            "cpu" ou "cuda" (défaut : cpu)
--load-checkpoint   Chemin d'un checkpoint .pth pour reprendre
--eval-episodes     Nombre d'épisodes d'évaluation (défaut : 3)
--render-eval       Afficher le rendu pendant l'évaluation
```

Reprendre depuis un checkpoint :

```bash
python vdn.py --load-checkpoint checkpoints/vdn_ice_hockey/modele_vdn_ep_500.pth
```

### `vdn_amélioré.py` — Version optimisée GPU

Ajoute trois fonctionnalités par rapport à la baseline :

- **Mixed precision (AMP)** via `torch.amp` — entraînement plus rapide sur GPU
- **Reward shaping** — bonus de récompense pour tirer (+0.01) et se déplacer (+0.001)
- **Multi-environnements** — collecte d'expériences sur N instances en parallèle

```bash
cd VDN
python vdn_amélioré.py
```

Options disponibles :

```
--episodes    Nombre d'épisodes (défaut : 500)
--num-envs    Nombre d'environnements parallèles (défaut : 1)
```

Les autres hyperparamètres (lr, gamma, AMP, reward shaping...) sont à modifier directement dans la classe `TrainConfig` en tête de fichier.

### Visualiser les courbes (TensorBoard)

```bash
tensorboard --logdir VDN/runs/
```

### Sur Google Colab

Ouvrir `VDN_lancement.ipynb` dans Colab et suivre les cellules dans l'ordre :
1. Monter le Google Drive
2. Installer les dépendances
3. Lancer l'entraînement via `xvfb-run -a python vdn_amélioré.py ...` (affichage virtuel requis sur Colab)

---

## Notes d'architecture

### MA-POCA

L'algorithme suit le pseudo-code de l'article *"Intelligent Close Air Combat Design based on MA-POCA Algorithm"* :

1. **Encodeur d'observation** (`observation_encoding.py`) : Impala CNN (Conv-ReLU + blocs résiduels) adapté aux images ATARI de petite taille, plus léger que ResNet/ImageNet
2. **Critique centralisé** (`centralized_critic.py`) : observe l'état global et les actions de tous les agents
3. **Mécanisme RSA** (`RSA.py`) : Relational Self-Attention pour modéliser les interactions entre agents
4. **Acteurs décentralisés** (`actor.py`) : chaque agent prend ses décisions depuis son observation locale uniquement

### VDN

VDN décompose la Q-valeur conjointe \( Q_{tot} \) en somme des Q-valeurs individuelles :

$$Q_{tot}(s, \mathbf{a}) = \sum_{i} Q_i(o_i, a_i)$$

Ce qui permet d'entraîner les agents de façon centralisée tout en exécutant les politiques de façon décentralisée (paradigme CTDE).
