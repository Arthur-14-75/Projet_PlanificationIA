# baseline VDN  pour Ice Hockey de PettingZoo
from __future__ import annotations
# Import standard libraries
import argparse
import importlib
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import supersuit as ss
from pettingzoo.atari import ice_hockey_v2
from pettingzoo.atari.base_atari_env import BaseAtariEnv
from pettingzoo.utils.conversions import aec_to_parallel

# On définit les noms des agents et l'agent de base
AGENTS_LOGIQUES = ["agent_1","agent_2"]
AGENT_BASE = "first_0"

def creer_env(empilement_frames: int = 4, mode_rendu: str | None = None):
    # On prépare le jeu Ice Hockey d'Atari
    env_brut = BaseAtariEnv(
        game="ice_hockey",
        num_players=1,
        mode_num=None,
        env_name="ice_hockey_team_vdn",
        obs_type="grayscale_image",
        render_mode=mode_rendu,
    )
    # On transforme l'env pour qu'il soit gérable en parallèle
    env_parallele = aec_to_parallel(env_brut)
    # On redimensionne l'image en 84x84 pour que ce soit moins lourd
    env_parallele = ss.resize_v1(env_parallele, x_size=84,y_size=84)
    # On empile les images pour voir le mouvement
    env_parallele = ss.frame_stack_v1(env_parallele,empilement_frames)
    return WrapperEquipeVsOrdi(env_parallele)

class WrapperEquipeVsOrdi:
    # Cette classe c'est pour faire croire qu'on a deux agents alors qu'il n'y en a qu'un seul vrai
    def __init__(self, env_parallele_base):
        self.env_base = env_parallele_base
        self.agents_possibles = AGENTS_LOGIQUES.copy()
        self.agents = self.agents_possibles.copy()

    def reset(self, seed: int | None = None, options=None):
        obs_dict, infos = self.env_base.reset(seed=seed,options=options)
        obs_base = obs_dict[AGENT_BASE]
        info_base = infos.get(AGENT_BASE,{}) if isinstance(infos, dict) else {}
        self.agents = self.agents_possibles.copy()
        # On donne la même image aux deux agents
        return (
            {"agent_1": obs_base, "agent_2": obs_base},
            {"agent_1": info_base, "agent_2": info_base},
        )

    def step(self, actions: Dict[str, int]):
        # On prend l'action de l'agent 1 pour le joueur réel
        action_reelle = int(actions["agent_1"])
        obs_dict,recompenses,termines,tronques,infos = self.env_base.step({AGENT_BASE: action_reelle})

        obs_base=obs_dict[AGENT_BASE]
        recompense_base=float(recompenses[AGENT_BASE])
        termine_base =bool(termines[AGENT_BASE])
        tronque_base = bool(tronques[AGENT_BASE])
        info_base=infos.get(AGENT_BASE, {}) if isinstance(infos, dict) else {}

        # On duplique tout pour nos deux agents fictifs
        obs={"agent_1": obs_base,"agent_2": obs_base}
        recompense = {"agent_1": recompense_base,"agent_2": recompense_base}
        termine={"agent_1": termine_base,"agent_2": termine_base}
        tronque ={"agent_1": tronque_base,"agent_2": tronque_base}
        info= {"agent_1": info_base,"agent_2": info_base}

        if termine_base or tronque_base:
            self.agents = []
        else:
            self.agents = self.agents_possibles.copy()
            
        return obs, recompense, termine, tronque, info

    def action_space(self, agent: str):
        return self.env_base.action_space(AGENT_BASE)

    def observation_space(self, agent: str):
        return self.env_base.observation_space(AGENT_BASE)

    def close(self):
        self.env_base.close()

    def render(self):
        return self.env_base.render()

def obs_en_chw(observation: np.ndarray) -> np.ndarray:
    # On change le format de l'image pour PyTorch 
    tableau = np.asarray(observation, dtype=np.uint8)
    if tableau.ndim == 2:
        tableau = tableau[..., None]
    return np.transpose(tableau,(2, 0, 1))

class ReseauQ(nn.Module):
    # Le cerveau de notre IA avec des couches de convolution
    def __init__(self, forme_obs: Tuple[int, int, int], n_actions: int):
        super().__init__()
        c, h, w = forme_obs
        self.convolutions = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Calcul de la taille de sortie des convs
        with torch.no_grad():
            n_plat = self.convolutions(torch.zeros(1, c, h, w)).view(1, -1).shape[1]
        
        self.couches_finales = nn.Sequential(
            nn.Linear(n_plat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(x)
        x = x.reshape(x.size(0), -1)
        return self.couches_finales(x)

class MemoireReplay:
    # Pour stocker les souvenirs des parties passées
    def __init__(self, capacite: int, forme_obs: Tuple[int, int, int], n_agents: int):
        self.capacite = capacite
        self.toutes_obs = torch.empty((capacite + 1, *forme_obs), dtype=torch.uint8)
        self.toutes_actions = torch.empty((capacite, n_agents), dtype=torch.int16)
        self.toutes_recompenses = torch.empty((capacite, n_agents), dtype=torch.float32)
        self.tous_finis = torch.empty((capacite,), dtype=torch.uint8)
        self.debut = 0
        self.debut_obs = 0
        self.position = 0
        self.taille_actuelle = 0

    def ajouter(self, obs, actions, recompenses, fini):
        if self.taille_actuelle < self.capacite:
            self.position = (self.debut + self.taille_actuelle) % self.capacite
            pos_obs = (self.debut_obs + self.taille_actuelle) % (self.capacite + 1)
            self.taille_actuelle += 1
        else:
            self.debut = (self.debut + 1) % self.capacite
            self.debut_obs=(self.debut_obs + 1) % (self.capacite + 1)
            self.position=(self.debut + self.taille_actuelle - 1) % self.capacite
            pos_obs = (self.debut_obs + self.taille_actuelle - 1) % (self.capacite + 1)

        self.toutes_obs[pos_obs].copy_(torch.as_tensor(obs, dtype=torch.uint8))
        self.toutes_actions[self.position].copy_(torch.as_tensor(actions, dtype=torch.int16))
        self.toutes_recompenses[self.position].copy_(torch.as_tensor(recompenses, dtype=torch.float32))
        self.tous_finis[self.position] = 1 if fini else 0

    def echantillonner(self, taille_batch: int):
        indices = torch.randint(0, self.taille_actuelle, (taille_batch,))
        indices_trans = (self.debut + indices) % self.capacite
        indices_obs=(self.debut_obs + indices) % (self.capacite + 1)
        indices_obs_suivantes = (indices_obs + 1) % (self.capacite + 1)
        return (
            self.toutes_obs[indices_obs],
            self.toutes_actions[indices_trans],
            self.toutes_recompenses[indices_trans],
            self.toutes_obs[indices_obs_suivantes],
            self.tous_finis[indices_trans].to(dtype=torch.float32),
        )

    def __len__(self) -> int:
        return self.taille_actuelle

@dataclass
class ConfigEntrainement:
    # Tous les réglages pour l'entrainement
    episodes: int = 500
    max_etapes: int = 1000
    gamma: float = 0.99
    lr: float = 1e-4
    taille_batch: int = 32
    taille_buffer: int = 50_000
    debut_apprentissage: int = 1_000
    frequence_train: int = 4
    maj_cible_freq: int = 1_000
    eps_debut: float = 1.0
    eps_fin: float = 0.05
    etapes_decroissance_eps: int = 100_000
    seed: int = 42
    device: str = "cpu"
    dossier_checkpoint: str = "checkpoints/vdn_ice_hockey"
    sauvegarde_chaque: int = 500
    tensorboard_dir: str = "runs/vdn_hockey_exp_1"
    charger_checkpoint: str | None = None

def fixer_seed(graine: int):
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)

def sauvegarder_modele(modele: ReseauQ, episode: int, dossier: str):
    Path(dossier).mkdir(parents=True, exist_ok=True)
    chemin = os.path.join(dossier, f"modele_vdn_ep_{episode}.pth")
    torch.save(modele.state_dict(), chemin)
    print(f"Modèle sauvegardé : {chemin}")

def preparer_obs_agents(obs_dict: Dict[str, np.ndarray], liste_agents: List[str]) -> np.ndarray:
    return np.stack([obs_en_chw(obs_dict[a]) for a in liste_agents], axis=0)

def preparer_obs_equipe(obs_dict: Dict[str, np.ndarray], liste_agents: List[str]) -> np.ndarray:
    # Comme les agents voient la même chose, on n'en garde qu'une
    return obs_en_chw(obs_dict[liste_agents[0]])

def calculer_recompenses_coop(recompenses_dict: Dict[str, float]) -> Dict[str, float]:
    valeur_commune = float(recompenses_dict["agent_1"])
    return {"agent_1": valeur_commune, "agent_2": valeur_commune}

def choisir_actions(modele: ReseauQ, obs_agents: np.ndarray, epsilon: float, n_actions: int, machine: torch.device) -> np.ndarray:
    # Stratégie epsilon-greedy pour l'exploration
    if random.random() < epsilon:
        return np.asarray([random.randrange(n_actions) for _ in range(obs_agents.shape[0])], dtype=np.int64)

    obs_t = torch.as_tensor(obs_agents, dtype=torch.float32, device=machine) / 255.0
    with torch.no_grad():
        valeurs_q = modele(obs_t)
        choix = torch.argmax(valeurs_q, dim=1)
    return choix.cpu().numpy().astype(np.int64)

def mise_a_jour_vdn(modele: ReseauQ, modele_cible: ReseauQ, optimiseur: optim.Optimizer, buffer: MemoireReplay, cfg: ConfigEntrainement, n_agents: int, machine: torch.device) -> float:
    # C'est ici qu'on fait l'apprentissage 
    obs, actions, recompenses, obs_suiv, finis = buffer.echantillonner(cfg.taille_batch)

    obs_t=obs.to(device=machine, dtype=torch.float32) / 255.0
    actions_t=actions.to(device=machine, dtype=torch.int64)
    recompenses_t = recompenses.to(device=machine, dtype=torch.float32)
    obs_suiv_t=obs_suiv.to(device=machine, dtype=torch.float32) / 255.0
    finis_t = finis.to(device=machine, dtype=torch.float32)

    taille = obs_t.shape[0]
    obs_t = obs_t.unsqueeze(1).expand(-1, n_agents, -1, -1, -1)
    obs_suiv_t = obs_suiv_t.unsqueeze(1).expand(-1, n_agents, -1, -1, -1)
    
    obs_plat = obs_t.reshape(taille * n_agents, *obs_t.shape[2:])
    obs_suiv_plat = obs_suiv_t.reshape(taille * n_agents, *obs_suiv_t.shape[2:])

    q_tous=modele(obs_plat).reshape(taille, n_agents, -1)
    q_pris=torch.gather(q_tous, dim=2, index=actions_t.unsqueeze(-1)).squeeze(-1)
    q_total = q_pris.sum(dim=1) 
    with torch.no_grad():
        next_q_tous =modele_cible(obs_suiv_plat).reshape(taille, n_agents, -1)
        next_q_max =next_q_tous.max(dim=2).values
        next_q_total= next_q_max.sum(dim=1)
        recompense_totale =recompenses_t.sum(dim=1)
        cible = recompense_totale + (1.0 - finis_t) * cfg.gamma * next_q_total

    perte = F.mse_loss(q_total, cible)
    optimiseur.zero_grad()
    perte.backward()
    nn.utils.clip_grad_norm_(modele.parameters(), 10.0)
    optimiseur.step()
    return float(perte.item())

def entrainer_ia(cfg: ConfigEntrainement):
    fixer_seed(cfg.seed)
    machine = torch.device(cfg.device)

    env = creer_env(empilement_frames=4)
    liste_agents = list(env.agents_possibles)
    n_agents = len(liste_agents)
    
    obs_dict, _=env.reset(seed=cfg.seed)
    obs_exemple=obs_en_chw(obs_dict[liste_agents[0]])
    forme_obs=obs_exemple.shape
    n_actions = env.action_space(liste_agents[0]).n

    modele = ReseauQ(forme_obs=forme_obs, n_actions=n_actions).to(machine)
    modele_cible = ReseauQ(forme_obs=forme_obs, n_actions=n_actions).to(machine)
    modele_cible.load_state_dict(modele.state_dict())

    optimiseur=optim.Adam(modele.parameters(), lr=cfg.lr)
    buffer = MemoireReplay(cfg.taille_buffer, forme_obs=forme_obs, n_agents=n_agents)
    journaliste=SummaryWriter(cfg.tensorboard_dir)

    etape_globale = 0
    epsilon=cfg.eps_debut
    fenetre_scores = deque(maxlen=20)
    ep_depart = 1

    # On charge un ancien fichier si besoin 
    if cfg.charger_checkpoint:
        try:
            modele.load_state_dict(torch.load(cfg.charger_checkpoint, map_location=machine))
            if "ep_" in cfg.charger_checkpoint:
                ep_depart = int(cfg.charger_checkpoint.split("ep_")[1].split(".")[0]) + 1
        except Exception as e:
            print(f"Erreur de chargement : {e}")
            return None, liste_agents

    for ep in range(ep_depart, cfg.episodes + 1):
        obs_dict, _ = env.reset(seed=cfg.seed + ep)
        score_ep = 0.0
        scores_par_agent = np.zeros(n_agents, dtype=np.float32)
        liste_pertes = []

        for _ in range(cfg.max_etapes):
            obs_pour_choix = preparer_obs_agents(obs_dict, liste_agents)
            act_tableau=choisir_actions(modele, obs_pour_choix, epsilon, n_actions, machine)
            dico_actions={a: int(act_tableau[i]) for i, a in enumerate(liste_agents)}

            suiv_obs_dict, dico_recomp, termines, tronques, _ = env.step(dico_actions)
            est_fini = all(bool(termines[a] or tronques[a]) for a in liste_agents)

            recomp_coop=calculer_recompenses_coop(dico_recomp)
            recomp_tableau = np.asarray([recomp_coop[a] for a in liste_agents], dtype=np.float32)
            obs_simple=preparer_obs_equipe(obs_dict, liste_agents)

            buffer.ajouter(obs_simple, act_tableau, recomp_tableau, est_fini)

            score_ep += float(recomp_tableau.sum())
            scores_par_agent += recomp_tableau
            obs_dict = suiv_obs_dict
            etape_globale += 1

            if len(buffer) >= cfg.debut_apprentissage and etape_globale % cfg.frequence_train == 0:
                p = mise_a_jour_vdn(modele, modele_cible, optimiseur, buffer, cfg, n_agents, machine)
                liste_pertes.append(p)

            if etape_globale % cfg.maj_cible_freq == 0:
                modele_cible.load_state_dict(modele.state_dict())

            # On diminue epsilon petit à petit
            progression = min(1.0, etape_globale / float(cfg.etapes_decroissance_eps))
            epsilon = cfg.eps_debut + progression * (cfg.eps_fin - cfg.eps_debut)

            if est_fini:
                break

        fenetre_scores.append(score_ep)
        moyenne_score=float(np.mean(fenetre_scores))
        moyenne_perte=float(np.mean(liste_pertes)) if liste_pertes else 0.0
        
        print(f"Episode {ep:4d} | score={score_ep:8.2f} | moyenne={moyenne_score:8.2f} | eps={epsilon:5.3f}")
        
        journaliste.add_scalar("Entrainement/Perte", moyenne_perte, ep)
        journaliste.add_scalar("Entrainement/Score", moyenne_score, ep)

        if cfg.sauvegarde_chaque > 0 and ep % cfg.sauvegarde_chaque == 0:
            sauvegarder_modele(modele, ep, cfg.dossier_checkpoint)

    journaliste.close()
    env.close()
    return modele, liste_agents

def evaluer_ia(modele: ReseauQ, liste_agents: List[str], episodes: int = 5, max_etapes: int = 2000, rendu: bool = False):
    machine = next(modele.parameters()).device
    mode_rendu = "human" if rendu else None
    env = creer_env(empilement_frames=4, mode_rendu=mode_rendu)
    
    scores = []
    victoires, defaites, nuls = 0, 0, 0

    for ep in range(episodes):
        obs_dict, _=env.reset(seed=10_000 + ep)
        score_total = 0.0

        for _ in range(max_etapes):
            obs_pour_choix = preparer_obs_agents(obs_dict, liste_agents)
            act_tableau = choisir_actions(modele, obs_pour_choix, 0.0, env.action_space(liste_agents[0]).n, machine)
            dico_actions = {a: int(act_tableau[i]) for i, a in enumerate(liste_agents)}
            
            obs_dict, dico_recomp, termines, tronques, _ = env.step(dico_actions)
            score_total += float(sum(dico_recomp.values()))

            if all(bool(termines[a] or tronques[a]) for a in liste_agents):
                break

        scores.append(score_total)
        if score_total > 0: victoires += 1
        elif score_total < 0: defaites += 1
        else: nuls += 1
        print(f"Eval episode {ep + 1}: score={score_total:.2f}")

    env.close()
    print(f"Moyenne Eval: {np.mean(scores):.2f}")
    print(f"Stats: V:{victoires}, D:{defaites}, N:{nuls}")

def lire_arguments():
    # Sert à lire les trucs qu'on écrit dans le terminal
    parser = argparse.ArgumentParser(description="IA pour le Hockey sur Atari")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load-checkpoint", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--render-eval", action="store_true")

    args = parser.parse_args()
    config = ConfigEntrainement(
        episodes=args.episodes,
        max_steps=args.max_steps,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        charger_checkpoint=args.load_checkpoint,
    )
    return config, args.eval_episodes, args.render_eval

if __name__ == "__main__":
    ma_config, nb_eval, faire_rendu = lire_arguments()
    mon_modele, mes_agents = entrainer_ia(ma_config)
    
    if mon_modele is not None:
        evaluer_ia(mon_modele, mes_agents, episodes=nb_eval, rendu=faire_rendu)