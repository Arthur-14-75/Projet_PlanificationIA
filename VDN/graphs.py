import matplotlib.pyplot as plt 
import re

episodes = []
joint_returns = []
mean_returns = []

# Ouvre ton fichier texte (assure-toi que le nom correspond)
with open('mes_logs.txt', 'r') as file:
    for line in file:
        # On ne cherche que les lignes qui contiennent les infos d'épisode
        if "Episode" in line and "joint_return=" in line:
            # Cette formule (Regex) extrait automatiquement les nombres de ta ligne de log
            match = re.search(r'Episode\s+(\d+)\s+\|\s+joint_return=\s+([\d.-]+)\s+\|\s+mean_joint_return=\s+([\d.-]+)', line)
            
            if match:
                episodes.append(int(match.group(1)))
                joint_returns.append(float(match.group(2)))
                mean_returns.append(float(match.group(3)))

# --- Création du graphique ---
plt.figure(figsize=(12, 6))

# 1. On trace les scores de chaque épisode (en gris et transparent pour ne pas surcharger)
plt.plot(episodes, joint_returns, color='lightgray', alpha=0.5, label="Score exact de l'épisode")

# 2. On trace ta moyenne (la vraie courbe d'apprentissage, bien visible)
plt.plot(episodes, mean_returns, color='red', linewidth=2.5, label="Moyenne (mean_joint_return)")

# --- Décoration du graphique ---
plt.title("Courbe d'apprentissage de l'IA (Atari Ice Hockey)", fontsize=14, fontweight='bold')
plt.xlabel("Nombre d'épisodes", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc="upper left")

# Affiche le tout
plt.show()