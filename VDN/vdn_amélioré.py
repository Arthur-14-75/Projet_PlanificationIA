from __future__ import annotations
# Importations de bibliotheques
import argparse
import gc
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
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import supersuit as ss
from pettingzoo.atari import ice_hockey_v2
from pettingzoo.atari.base_atari_env import BaseAtariEnv
from pettingzoo.utils.conversions import aec_to_parallel

# les noms des joueurs
LOGICAL_AGENTS=["agent_1","agent_2"]
BASE_AGENT = "first_0"

def make_env(frame_stack: int= 4, render_mode: str | None=None):
    # On cree le jeu de hockey ici
    raw_single_player = BaseAtariEnv(
        game="ice_hockey",
        num_players=1,
        mode_num=None,
        env_name="ice_hockey_team_vdn",
        obs_type="grayscale_image",
        render_mode=render_mode,
    )
    parallel_env =aec_to_parallel(raw_single_player)
    parallel_env =ss.resize_v1(parallel_env, x_size=84,y_size=84)
    parallel_env = ss.frame_stack_v1(parallel_env,frame_stack)
    return TeamVsComputerWrapper(parallel_env)

class TeamVsComputerWrapper:
    # un truc pour faire croire qu'on est deux
    def __init__(self, base_parallel_env):
        self.base_env=base_parallel_env
        self.possible_agents = LOGICAL_AGENTS.copy()
        self.agents= self.possible_agents.copy()

    def reset(self, seed: int | None= None, options=None):
        # on remet le jeu a zero
        obs_dict, infos = self.base_env.reset(seed=seed,options=options)
        base_obs =obs_dict[BASE_AGENT]
        base_info=infos.get(BASE_AGENT, {}) if isinstance(infos, dict) else {}
        self.agents = self.possible_agents.copy()
        return (
            {"agent_1": base_obs,"agent_2": base_obs},
            {"agent_1": base_info,"agent_2": base_info},
        )

    def step(self, actions: Dict[str, int]):
        # on fait bouger le joueur 1
        action= int(actions["agent_1"])
        obs_dict, rewards, terms, truncs, infos = self.base_env.step({BASE_AGENT: action})
        base_obs= obs_dict[BASE_AGENT]
        base_reward = float(rewards[BASE_AGENT])
        base_term= bool(terms[BASE_AGENT])
        base_trunc= bool(truncs[BASE_AGENT])
        base_info =infos.get(BASE_AGENT, {}) if isinstance(infos, dict) else {}
        obs= {"agent_1": base_obs,"agent_2": base_obs}
        reward = {"agent_1": base_reward,"agent_2": base_reward}
        term= {"agent_1": base_term,"agent_2": base_term}
        trunc={"agent_1": base_trunc,"agent_2": base_trunc}
        info = {"agent_1": base_info,"agent_2": base_info}
        self.agents = [] if (base_term or base_trunc) else self.possible_agents.copy()
        return obs,reward,term,trunc,info

    def action_space(self, agent: str):
        return self.base_env.action_space(BASE_AGENT)
    def observation_space(self, agent: str):
        return self.base_env.observation_space(BASE_AGENT)
    def close(self):
        self.base_env.close()
    def render(self):
        return self.base_env.render()

def obs_to_chw(obs: np.ndarray) -> np.ndarray:
    # change la forme de l'image
    arr = np.asarray(obs, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None]
    return np.transpose(arr, (2, 0, 1))

class MultiEnvWrapper:
    # pour lancer plusieurs jeux en meme temps
    def __init__(self, num_envs: int= 4):
        self.num_envs = num_envs
        self.envs=[make_env() for _ in range(num_envs)]
        self.agents = list(self.envs[0].possible_agents)
        self.obs_dicts = [None] * num_envs
        self.reset_all()
    
    def reset_all(self, seed: int= None):
        self.obs_dicts = []
        for i, env in enumerate(self.envs):
            obs_dict, _ = env.reset(seed=seed + i if seed is not None else None)
            self.obs_dicts.append(obs_dict)
    
    def step_all(self, action_dicts: List[Dict[str, int]]):
        obs_list = []
        reward_list = []
        done_list = []
        for env_idx, (env, action_dict) in enumerate(zip(self.envs, action_dicts)):
            obs_dict, rewards_dict, terms, truncs, _ = env.step(action_dict)
            done=all(bool(terms[a] or truncs[a]) for a in self.agents)
            obs_list.append(obs_dict)
            reward_list.append(rewards_dict)
            done_list.append(done)
            if done:
                obs_dict, _ = env.reset()
                obs_list[env_idx] = obs_dict
        self.obs_dicts = obs_list
        return obs_list,reward_list,done_list
    
    def close(self):
        for env in self.envs: env.close()

def pack_multi_env_obs(obs_dicts: List[Dict], agents: List[str]) -> np.ndarray:
    return np.stack([np.stack([obs_to_chw(obs_dicts[i][a]) for a in agents], axis=0) for i in range(len(obs_dicts))], axis=0)

class QNet(nn.Module):
    # mon reseau de neurones
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int, hidden_size: int= 512):
        super().__init__()
        c, h, w = obs_shape
        self.conv=nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_flat, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)

class ReplayBuffer:
    # la ou on garde les souvenirs
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], n_agents: int):
        self.capacity =capacity
        self.obs=torch.empty((capacity + 1, *obs_shape), dtype=torch.uint8)
        self.actions = torch.empty((capacity, n_agents), dtype=torch.int16)
        self.rewards =torch.empty((capacity, n_agents), dtype=torch.float32)
        self.dones =torch.empty((capacity,), dtype=torch.uint8)
        self.start =0
        self.obs_start =0
        self.pos =0
        self.size = 0

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, done: bool):
        # ajoute un nouveau souvenir
        if self.size < self.capacity:
            self.pos =(self.start + self.size) % self.capacity
            obs_pos =(self.obs_start + self.size) % (self.capacity + 1)
            self.size += 1
        else:
            self.start = (self.start + 1) % self.capacity
            self.obs_start = (self.obs_start + 1) % (self.capacity + 1)
            self.pos =(self.start + self.size - 1) % self.capacity
            obs_pos =(self.obs_start + self.size - 1) % (self.capacity + 1)
        self.obs[obs_pos].copy_(torch.as_tensor(obs, dtype=torch.uint8))
        self.actions[self.pos].copy_(torch.as_tensor(actions, dtype=torch.int16))
        self.rewards[self.pos].copy_(torch.as_tensor(rewards, dtype=torch.float32))
        self.dones[self.pos] = 1 if done else 0

    def sample(self, batch_size: int):
        # pioche des souvenirs au hasard
        indices = torch.randint(0, self.size, (batch_size,))
        trans_indices = (self.start + indices) % self.capacity
        obs_indices = (self.obs_start + indices) % (self.capacity + 1)
        next_obs_indices = (obs_indices + 1) % (self.capacity + 1)
        return (self.obs[obs_indices], self.actions[trans_indices], self.rewards[trans_indices], self.obs[next_obs_indices], self.dones[trans_indices].to(dtype=torch.float32))

    def __len__(self) -> int: return self.size

@dataclass
class TrainConfig:
    # plein de chiffres pour l'entrainement
    episodes: int= 500
    max_steps: int= 1000
    gamma: float= 0.99
    lr: float= 1e-4
    batch_size: int= 32
    buffer_size: int= 50000
    learning_starts: int= 1000
    train_freq: int= 4
    target_update_freq: int= 1000
    eps_start: float= 1.0
    eps_end: float= 0.05
    eps_decay_steps: int= 100000
    seed: int= 42
    device: str= "cpu"
    checkpoint_dir: str= "checkpoints/vdn_ice_hockey"
    checkpoint_every: int= 500
    tensorboard_logdir: str= "runs/vdn_hockey_exp_1"
    load_checkpoint: str | None= None
    reward_shaping_enabled: bool= True
    reward_shaping_fire: float= 0.01
    reward_shaping_move: float= 0.001
    use_amp: bool= True
    qnet_hidden_size: int= 512
    num_envs: int= 1

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(model: QNet, episode: int, checkpoint_dir: str):
    # sauvegarde le modele
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"modele_vdn_ep_{episode}.pth")
    torch.save(model.state_dict(), path)
    print(f"Sauvegarde ok: {path}")

def pack_agent_obs(obs_dict: Dict[str, np.ndarray], agents: List[str]) -> np.ndarray:
    return np.stack([obs_to_chw(obs_dict[a]) for a in agents], axis=0)

def pack_team_obs(obs_dict: Dict[str, np.ndarray], agents: List[str]) -> np.ndarray:
    return obs_to_chw(obs_dict[agents[0]])

def compute_reward_shaping(actions: np.ndarray, reward_shaping_enabled: bool= True, fire_bonus: float= 0.01, move_bonus: float= 0.001) -> float:
    # donne des petits points bonus pour bouger
    if not reward_shaping_enabled: return 0.0
    shaping_bonus = 0.0
    FIRE_ACTIONS = {1, 3, 5}
    MOVEMENT_ACTIONS = {2, 3, 4, 5, 6, 7}
    for action in actions:
        action_id = int(action)
        if action_id in FIRE_ACTIONS: shaping_bonus += fire_bonus
        elif action_id in MOVEMENT_ACTIONS: shaping_bonus += move_bonus
    return shaping_bonus

def build_coop_rewards(rewards_dict: Dict[str, float], actions: np.ndarray= None, reward_shaping_enabled: bool= True, fire_bonus: float= 0.01, move_bonus: float= 0.001) -> Dict[str, float]:
    # met les recompenses en commun
    team_reward = float(rewards_dict["agent_1"])
    if actions is not None:
        shaping_bonus =compute_reward_shaping(actions, reward_shaping_enabled, fire_bonus, move_bonus)
        team_reward += shaping_bonus
    return {"agent_1": team_reward,"agent_2": team_reward}

def select_actions(q_net: QNet, obs_agents: np.ndarray, epsilon: float, n_actions: int, device: torch.device) -> np.ndarray:
    # choisit quoi faire (hasard ou reflechi)
    if random.random() < epsilon:
        return np.asarray([random.randrange(n_actions) for _ in range(obs_agents.shape[0])], dtype=np.int64)
    obs_t =torch.as_tensor(obs_agents, dtype=torch.float32, device=device) / 255.0
    with torch.no_grad():
        q_values = q_net(obs_t)
        actions=torch.argmax(q_values,  dim=1)
    return actions.cpu().numpy().astype(np.int64)

def vdn_update(q_net: QNet, target_net: QNet, optimizer: optim.Optimizer, buffer: ReplayBuffer, cfg: TrainConfig, n_agents: int, device: torch.device, scaler: GradScaler | None= None) -> float:
    # c'est la qu'on apprend vraiment
    obs, actions, rewards, next_obs, dones = buffer.sample(cfg.batch_size)
    obs_t=obs.to(device=device, dtype=torch.float32) / 255.0
    actions_t=actions.to(device=device, dtype=torch.int64)
    rewards_t=rewards.to(device=device, dtype=torch.float32)
    next_obs_t=next_obs.to(device=device, dtype=torch.float32) / 255.0
    dones_t = dones.to(device=device, dtype=torch.float32)
    bsz = obs_t.shape[0]
    obs_t = obs_t.unsqueeze(1).expand(-1, n_agents, -1, -1, -1)
    next_obs_t =next_obs_t.unsqueeze(1).expand(-1, n_agents, -1, -1, -1)
    obs_flat = obs_t.reshape(bsz * n_agents, *obs_t.shape[2:])
    next_obs_flat = next_obs_t.reshape(bsz * n_agents, *next_obs_t.shape[2:])
    
    dev_str = "cuda" if "cuda" in str(device) else "cpu"
    with autocast(device_type=dev_str, enabled=cfg.use_amp):
        q_all=q_net(obs_flat).reshape(bsz, n_agents, -1)
        q_taken=torch.gather(q_all, dim=2, index=actions_t.unsqueeze(-1)).squeeze(-1)
        q_joint=q_taken.sum(dim=1)
        with torch.no_grad():
            next_q_all= target_net(next_obs_flat).reshape(bsz, n_agents, -1)
            next_q_max= next_q_all.max(dim=2).values
            next_q_joint= next_q_max.sum(dim=1)
            reward_joint =rewards_t.sum(dim=1)
            target=reward_joint + (1.0 - dones_t) * cfg.gamma * next_q_joint
        loss=F.mse_loss(q_joint, target)

    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()
    return float(loss.item())

def train_vdn(cfg: TrainConfig):
    # la boucle d'entrainement
    set_seed(cfg.seed)
    device =torch.device(cfg.device)
    print("Debut de l'entrainement hockey...")
    env = make_env(frame_stack=4)
    agents=list(env.possible_agents)
    n_agents=len(agents)
    obs_dict, _=env.reset(seed=cfg.seed)
    sample_obs=obs_to_chw(obs_dict[agents[0]])
    obs_shape=sample_obs.shape
    n_actions= env.action_space(agents[0]).n
    q_net =QNet(obs_shape=obs_shape, n_actions=n_actions, hidden_size=cfg.qnet_hidden_size).to(device)
    target_net =QNet(obs_shape=obs_shape, n_actions=n_actions, hidden_size=cfg.qnet_hidden_size).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer=optim.Adam(q_net.parameters(), lr=cfg.lr)
    scaler =GradScaler("cuda") if cfg.use_amp and "cuda" in str(device) else None
    buffer=ReplayBuffer(cfg.buffer_size, obs_shape=obs_shape, n_agents=n_agents)
    writer=SummaryWriter(cfg.tensorboard_logdir)
    global_step= 0
    epsilon=cfg.eps_start
    returns_window = deque(maxlen=20)
    
    if cfg.num_envs > 1:
        env_wrapper =MultiEnvWrapper(num_envs=cfg.num_envs)
        obs_dicts=env_wrapper.obs_dicts
        is_multi_env= True
    else:
        obs_dict, _ = env.reset(seed=cfg.seed)
        obs_dicts =[obs_dict]
        is_multi_env = False

    for ep in range(1, cfg.episodes + 1):
        if is_multi_env: env_wrapper.reset_all(seed=cfg.seed+ep)
        else: obs_dicts = [env.reset(seed=cfg.seed+ep)[0]]
        ep_return_raw = 0.0
        losses = deque(maxlen=100)
        for step in range(cfg.max_steps):
            act_arr_list = []
            for obs in obs_dicts:
                act_arr = select_actions(q_net, pack_agent_obs(obs, agents), epsilon, n_actions, device)
                act_arr_list.append(act_arr)
            action_dicts =[{a: int(act_arr_list[i][j]) for j, a in enumerate(agents)} for i in range(len(obs_dicts))]
            if is_multi_env: next_obs_dicts, rewards_dicts, dones = env_wrapper.step_all(action_dicts)
            else:
                n_o, r, t, tr, _= env.step(action_dicts[0])
                next_obs_dicts, rewards_dicts, dones = [n_o], [r], [all(t.values()) or all(tr.values())]
            for i in range(len(obs_dicts)):
                coop_r = build_coop_rewards(rewards_dicts[i], act_arr_list[i], cfg.reward_shaping_enabled, cfg.reward_shaping_fire, cfg.reward_shaping_move)
                buffer.add(pack_team_obs(obs_dicts[i], agents), act_arr_list[i], np.array([coop_r[a] for a in agents]), dones[i])
                ep_return_raw += sum(rewards_dicts[i].values())
            obs_dicts = next_obs_dicts
            global_step += len(obs_dicts)
            if len(buffer) >= cfg.learning_starts and global_step % cfg.train_freq == 0:
                losses.append(vdn_update(q_net, target_net, optimizer, buffer, cfg, n_agents, device, scaler))
            if global_step % cfg.target_update_freq == 0: target_net.load_state_dict(q_net.state_dict())
            epsilon = max(cfg.eps_end, cfg.eps_start - global_step / cfg.eps_decay_steps)
            if not is_multi_env and dones[0]: break
        avg_loss = np.mean(losses) if losses else 0
        print(f"Episode {ep} | Score: {ep_return_raw} | Loss: {avg_loss:.4f}")
        if ep % 10 == 0: gc.collect()
    writer.close()
    return q_net, agents

def evaluate(model: QNet, agents: List[str], episodes: int= 5, render: bool= False):
    # pour voir le resultat de l'entrainement en action
    env = make_env(render_mode="human" if render else None)
    for ep in range(episodes):
        obs_dict, _ = env.reset()
        done = False
        while not done:
            act_arr = select_actions(model, pack_agent_obs(obs_dict, agents), 0.0, env.action_space(agents[0]).n, next(model.parameters()).device)
            obs_dict, _, terms, truncs, _ = env.step({a: int(act_arr[i]) for i, a in enumerate(agents)})
            done = any(terms.values()) or any(truncs.values())
    env.close()

def parse_args():
    # pour passer des reglages en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()
    return TrainConfig(episodes=args.episodes,num_envs=args.num_envs), 3, False, False, "videos", 500

if __name__ == "__main__":
    # c'est la que ca commence
    c, e_ep, r_e, v_e, v_d, v_ey = parse_args()
    m, a = train_vdn(c)
    evaluate(m,a,episodes=e_ep)