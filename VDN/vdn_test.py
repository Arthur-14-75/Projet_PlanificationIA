"""
VDN baseline for PettingZoo Atari Ice Hockey.

Dependencies:
- pettingzoo[atari,accept-rom-license]
- supersuit
- torch
- numpy

Example:
python vdn_test.py --episodes 500 --batch-size 64
"""

from __future__ import annotations

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


LOGICAL_AGENTS = ["agent_1", "agent_2"]
BASE_AGENT = "first_0"


def make_env(frame_stack: int = 4, render_mode: str | None = None):
	raw_single_player = BaseAtariEnv(
		game="ice_hockey",
		num_players=1,
		mode_num=None,
		env_name="ice_hockey_team_vdn",
		obs_type="grayscale_image",
		render_mode=render_mode,
	)
	parallel_env = aec_to_parallel(raw_single_player)
	parallel_env = ss.resize_v1(parallel_env, x_size=84, y_size=84)
	parallel_env = ss.frame_stack_v1(parallel_env, frame_stack)
	return TeamVsComputerWrapper(parallel_env)


class TeamVsComputerWrapper:
	def __init__(self, base_parallel_env):
		self.base_env = base_parallel_env
		self.possible_agents = LOGICAL_AGENTS.copy()
		self.agents = self.possible_agents.copy()

	def reset(self, seed: int | None = None, options=None):
		obs_dict, infos = self.base_env.reset(seed=seed, options=options)
		base_obs = obs_dict[BASE_AGENT]
		base_info = infos.get(BASE_AGENT, {}) if isinstance(infos, dict) else {}
		self.agents = self.possible_agents.copy()
		return (
			{"agent_1": base_obs, "agent_2": base_obs},
			{"agent_1": base_info, "agent_2": base_info},
		)

	def step(self, actions: Dict[str, int]):
		# Both logical agents represent the same team; we apply agent_1 action to the real player.
		action = int(actions["agent_1"])
		obs_dict, rewards, terms, truncs, infos = self.base_env.step({BASE_AGENT: action})

		base_obs = obs_dict[BASE_AGENT]
		base_reward = float(rewards[BASE_AGENT])
		base_term = bool(terms[BASE_AGENT])
		base_trunc = bool(truncs[BASE_AGENT])
		base_info = infos.get(BASE_AGENT, {}) if isinstance(infos, dict) else {}

		obs = {"agent_1": base_obs, "agent_2": base_obs}
		reward = {"agent_1": base_reward, "agent_2": base_reward}
		term = {"agent_1": base_term, "agent_2": base_term}
		trunc = {"agent_1": base_trunc, "agent_2": base_trunc}
		info = {"agent_1": base_info, "agent_2": base_info}

		self.agents = [] if (base_term or base_trunc) else self.possible_agents.copy()
		return obs, reward, term, trunc, info

	def action_space(self, agent: str):
		return self.base_env.action_space(BASE_AGENT)

	def observation_space(self, agent: str):
		return self.base_env.observation_space(BASE_AGENT)

	def close(self):
		self.base_env.close()

	def render(self):
		return self.base_env.render()


def obs_to_chw(obs: np.ndarray) -> np.ndarray:
	arr = np.asarray(obs, dtype=np.uint8)
	if arr.ndim == 2:
		arr = arr[..., None]
	if arr.ndim != 3:
		raise ValueError(f"Unexpected observation shape: {arr.shape}")
	return np.transpose(arr, (2, 0, 1))


class QNet(nn.Module):
	def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
		super().__init__()
		c, h, w = obs_shape
		self.conv = nn.Sequential(
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
			nn.Linear(n_flat, 512),
			nn.ReLU(),
			nn.Linear(512, n_actions),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.conv(x)
		x = x.reshape(x.size(0), -1)
		return self.head(x)


class ReplayBuffer:
	def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], n_agents: int):
		self.capacity = capacity
		self.obs = torch.empty((capacity + 1, *obs_shape), dtype=torch.uint8)
		self.actions = torch.empty((capacity, n_agents), dtype=torch.int16)
		self.rewards = torch.empty((capacity, n_agents), dtype=torch.float32)
		self.dones = torch.empty((capacity,), dtype=torch.uint8)
		self.start = 0
		self.obs_start = 0
		self.pos = 0
		self.size = 0

	def add(
		self,
		obs: np.ndarray,
		actions: np.ndarray,
		rewards: np.ndarray,
		done: bool,
	):
		if self.size < self.capacity:
			self.pos = (self.start + self.size) % self.capacity
			obs_pos = (self.obs_start + self.size) % (self.capacity + 1)
			self.size += 1
		else:
			self.start = (self.start + 1) % self.capacity
			self.obs_start = (self.obs_start + 1) % (self.capacity + 1)
			self.pos = (self.start + self.size - 1) % self.capacity
			obs_pos = (self.obs_start + self.size - 1) % (self.capacity + 1)

		self.obs[obs_pos].copy_(torch.as_tensor(obs, dtype=torch.uint8))
		self.actions[self.pos].copy_(torch.as_tensor(actions, dtype=torch.int16))
		self.rewards[self.pos].copy_(torch.as_tensor(rewards, dtype=torch.float32))
		self.dones[self.pos] = 1 if done else 0

	def sample(self, batch_size: int):
		indices = torch.randint(0, self.size, (batch_size,))
		trans_indices = (self.start + indices) % self.capacity
		obs_indices = (self.obs_start + indices) % (self.capacity + 1)
		next_obs_indices = (obs_indices + 1) % (self.capacity + 1)
		return (
			self.obs[obs_indices],
			self.actions[trans_indices],
			self.rewards[trans_indices],
			self.obs[next_obs_indices],
			self.dones[trans_indices].to(dtype=torch.float32),
		)

	def __len__(self) -> int:
		return self.size


@dataclass
class TrainConfig:
	episodes: int = 500
	max_steps: int = 1000
	gamma: float = 0.99
	lr: float = 1e-4
	batch_size: int = 32
	buffer_size: int = 50_000
	learning_starts: int = 1_000
	train_freq: int = 4
	target_update_freq: int = 1_000
	eps_start: float = 1.0
	eps_end: float = 0.05
	eps_decay_steps: int = 100_000
	seed: int = 42
	device: str = "cpu"
	checkpoint_dir: str = "checkpoints/vdn_ice_hockey"
	checkpoint_every: int = 500
	tensorboard_logdir: str = "runs/vdn_hockey_exp_1"
	load_checkpoint: str | None = None


def set_seed(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def save_checkpoint(model: QNet, episode: int, checkpoint_dir: str):
	Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
	path = os.path.join(checkpoint_dir, f"modele_vdn_ep_{episode}.pth")
	torch.save(model.state_dict(), path)
	print(f"Checkpoint saved: {path}")


def pack_agent_obs(obs_dict: Dict[str, np.ndarray], agents: List[str]) -> np.ndarray:
	return np.stack([obs_to_chw(obs_dict[a]) for a in agents], axis=0)


def pack_team_obs(obs_dict: Dict[str, np.ndarray], agents: List[str]) -> np.ndarray:
	# Both agents see the same screen, so we keep only one copy for the replay buffer.
	return obs_to_chw(obs_dict[agents[0]])


def build_coop_rewards(rewards_dict: Dict[str, float]) -> Dict[str, float]:
	# Safety guard: enforce reward[agent_1] == reward[agent_2] even if wrappers change upstream.
	team_reward = float(rewards_dict["agent_1"])
	return {"agent_1": team_reward, "agent_2": team_reward}


def select_actions(
	q_net: QNet,
	obs_agents: np.ndarray,
	epsilon: float,
	n_actions: int,
	device: torch.device,
) -> np.ndarray:
	if random.random() < epsilon:
		return np.asarray([random.randrange(n_actions) for _ in range(obs_agents.shape[0])], dtype=np.int64)

	obs_t = torch.as_tensor(obs_agents, dtype=torch.float32, device=device)
	obs_t = obs_t / 255.0
	with torch.no_grad():
		q_values = q_net(obs_t)
		actions = torch.argmax(q_values, dim=1)
	return actions.cpu().numpy().astype(np.int64)


def vdn_update(
	q_net: QNet,
	target_net: QNet,
	optimizer: optim.Optimizer,
	buffer: ReplayBuffer,
	cfg: TrainConfig,
	n_agents: int,
	device: torch.device,
) -> float:
	obs, actions, rewards, next_obs, dones = buffer.sample(cfg.batch_size)

	obs_t = obs.to(device=device, dtype=torch.float32) / 255.0
	actions_t = actions.to(device=device, dtype=torch.int64)
	rewards_t = rewards.to(device=device, dtype=torch.float32)
	next_obs_t = next_obs.to(device=device, dtype=torch.float32) / 255.0
	dones_t = dones.to(device=device, dtype=torch.float32)

	bsz = obs_t.shape[0]
	obs_t = obs_t.unsqueeze(1).expand(-1, n_agents, -1, -1, -1)
	next_obs_t = next_obs_t.unsqueeze(1).expand(-1, n_agents, -1, -1, -1)
	obs_flat = obs_t.reshape(bsz * n_agents, *obs_t.shape[2:])
	next_obs_flat = next_obs_t.reshape(bsz * n_agents, *next_obs_t.shape[2:])

	q_all = q_net(obs_flat).reshape(bsz, n_agents, -1)
	q_taken = torch.gather(q_all, dim=2, index=actions_t.unsqueeze(-1)).squeeze(-1)
	q_joint = q_taken.sum(dim=1)

	with torch.no_grad():
		next_q_all = target_net(next_obs_flat).reshape(bsz, n_agents, -1)
		next_q_max = next_q_all.max(dim=2).values
		next_q_joint = next_q_max.sum(dim=1)
		reward_joint = rewards_t.sum(dim=1)
		target = reward_joint + (1.0 - dones_t) * cfg.gamma * next_q_joint

	loss = F.mse_loss(q_joint, target)
	optimizer.zero_grad()
	loss.backward()
	nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
	optimizer.step()
	return float(loss.item())


def train_vdn(cfg: TrainConfig):
	set_seed(cfg.seed)
	device = torch.device(cfg.device)

	env = make_env(frame_stack=4, render_mode=None)
	agents = list(env.possible_agents)
	n_agents = len(agents)
	if n_agents < 2:
		raise RuntimeError("Ice Hockey should provide at least 2 agents.")
	print("Environment mode: logical team of 2 agents vs built-in computer")
	print("Cooperative reward enabled: reward['agent_1'] == reward['agent_2']")

	obs_dict, _ = env.reset(seed=cfg.seed)
	sample_obs = obs_to_chw(obs_dict[agents[0]])
	obs_shape = sample_obs.shape
	n_actions = env.action_space(agents[0]).n

	q_net = QNet(obs_shape=obs_shape, n_actions=n_actions).to(device)
	target_net = QNet(obs_shape=obs_shape, n_actions=n_actions).to(device)
	target_net.load_state_dict(q_net.state_dict())

	optimizer = optim.Adam(q_net.parameters(), lr=cfg.lr)
	buffer = ReplayBuffer(cfg.buffer_size, obs_shape=obs_shape, n_agents=n_agents)
	writer = SummaryWriter(cfg.tensorboard_logdir)

	global_step = 0
	epsilon = cfg.eps_start
	returns_window = deque(maxlen=20)
	start_episode = 1

	if cfg.load_checkpoint:
		try:
			checkpoint_path = cfg.load_checkpoint
			q_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
			print(f"Loaded checkpoint from: {checkpoint_path}")
			if "ep_" in checkpoint_path:
				try:
					ep_num = int(checkpoint_path.split("ep_")[1].split(".")[0])
					start_episode = ep_num + 1
					print(f"Resuming from episode {start_episode}")
				except ValueError:
					pass
		except Exception as e:
			print(f"Failed to load checkpoint: {e}")
			return None, agents

	for ep in range(start_episode, cfg.episodes + 1):
		obs_dict, _ = env.reset(seed=cfg.seed + ep)
		ep_return_joint = 0.0
		ep_return_per_agent = np.zeros(n_agents, dtype=np.float32)
		losses = []

		for _ in range(cfg.max_steps):
			obs_agents = pack_agent_obs(obs_dict, agents)
			act_arr = select_actions(q_net, obs_agents, epsilon, n_actions, device)
			action_dict = {a: int(act_arr[i]) for i, a in enumerate(agents)}

			next_obs_dict, rewards_dict, terms, truncs, _ = env.step(action_dict)
			done = all(bool(terms[a] or truncs[a]) for a in agents)

			coop_rewards = build_coop_rewards(rewards_dict)
			rewards_arr = np.asarray([coop_rewards[a] for a in agents], dtype=np.float32)
			obs_single = pack_team_obs(obs_dict, agents)

			buffer.add(obs_single, act_arr, rewards_arr, done)

			ep_return_joint += float(rewards_arr.sum())
			ep_return_per_agent += rewards_arr

			obs_dict = next_obs_dict
			global_step += 1

			if len(buffer) >= cfg.learning_starts and global_step % cfg.train_freq == 0:
				loss = vdn_update(q_net, target_net, optimizer, buffer, cfg, n_agents, device)
				losses.append(loss)

			if global_step % cfg.target_update_freq == 0:
				target_net.load_state_dict(q_net.state_dict())

			frac = min(1.0, global_step / float(cfg.eps_decay_steps))
			epsilon = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

			if done:
				break

		returns_window.append(ep_return_joint)
		avg_return = float(np.mean(returns_window))
		avg_loss = float(np.mean(losses)) if losses else 0.0
		print(
			f"Episode {ep:4d} | joint_return={ep_return_joint:8.2f} "
			f"| mean_joint_return={avg_return:8.2f} | eps={epsilon:5.3f} | loss={avg_loss:8.4f} "
			f"| per_agent={ep_return_per_agent.tolist()}"
		)
		writer.add_scalar("Training/Loss", avg_loss, ep)
		writer.add_scalar("Training/Return", avg_return, ep)

		if cfg.checkpoint_every > 0 and ep % cfg.checkpoint_every == 0:
			save_checkpoint(model=q_net, episode=ep, checkpoint_dir=cfg.checkpoint_dir)

	writer.close()

	env.close()
	return q_net, agents


def evaluate(
	model: QNet,
	agents: List[str],
	episodes: int = 5,
	max_steps: int = 2000,
	render: bool = False,
	record_video: bool = False,
	video_dir: str = "videos/vdn_ice_hockey",
	video_every: int = 500,
	train_episode: int = 0,
):
	device = next(model.parameters()).device
	should_record = record_video and video_every > 0 and train_episode % video_every == 0
	render_mode = "human" if render else ("rgb_array" if should_record else None)
	env = make_env(frame_stack=4, render_mode=render_mode)
	returns = []
	victoires = 0
	defaites = 0
	egalites = 0
	video_writer = None
	if should_record:
		try:
			imageio = importlib.import_module("imageio.v2")

			Path(video_dir).mkdir(parents=True, exist_ok=True)
			video_path = os.path.join(video_dir, "eval_episode_1.mp4")
			video_writer = imageio.get_writer(video_path, fps=30)
			print(f"Video recording enabled: {video_path}")
		except ModuleNotFoundError:
			print("Video disabled: imageio is not installed (pip install imageio).")
			record_video = False
	elif record_video:
		print(f"Video skipped: train_episode={train_episode} is not a multiple of video_every={video_every}.")

	for ep in range(episodes):
		obs_dict, _ = env.reset(seed=10_000 + ep)
		ep_ret = 0.0

		for _ in range(max_steps):
			obs_agents = pack_agent_obs(obs_dict, agents)
			act_arr = select_actions(
				q_net=model,
				obs_agents=obs_agents,
				epsilon=0.0,
				n_actions=env.action_space(agents[0]).n,
				device=device,
			)
			action_dict = {a: int(act_arr[i]) for i, a in enumerate(agents)}
			obs_dict, rewards_dict, terms, truncs, _ = env.step(action_dict)
			if record_video and video_writer is not None and ep == 0:
				frame = env.render()
				if frame is not None:
					video_writer.append_data(frame)
			coop_rewards = build_coop_rewards(rewards_dict)
			ep_ret += float(sum(coop_rewards[a] for a in LOGICAL_AGENTS))

			done = all(bool(terms[a] or truncs[a]) for a in agents)
			if done:
				break

		returns.append(ep_ret)
		if ep_ret > 0:
			victoires += 1
		elif ep_ret < 0:
			defaites += 1
		else:
			egalites += 1
		print(f"Eval episode {ep + 1}: joint_return={ep_ret:.2f}")

	env.close()
	if video_writer is not None:
		video_writer.close()
	print(f"Eval mean joint return: {np.mean(returns):.2f}")
	win_rate = (100.0 * victoires / episodes) if episodes > 0 else 0.0
	print(f"Win rate: {win_rate:.1f}% | V: {victoires}, D: {defaites}, E: {egalites}")


def parse_args() -> Tuple[TrainConfig, int, bool, bool, str]:
	parser = argparse.ArgumentParser(description="Train a VDN baseline on PettingZoo Atari Ice Hockey")
	parser.add_argument("--episodes", type=int, default=500)
	parser.add_argument("--max-steps", type=int, default=1000)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--buffer-size", type=int, default=50_000)
	parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
	parser.add_argument("--learning-starts", type=int, default=1_000)
	parser.add_argument("--train-freq", type=int, default=4)
	parser.add_argument("--target-update-freq", type=int, default=1_000)
	parser.add_argument("--eps-start", type=float, default=1.0)
	parser.add_argument("--eps-end", type=float, default=0.05)
	parser.add_argument("--eps-decay-steps", type=int, default=100_000)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/vdn_ice_hockey")
	parser.add_argument("--checkpoint-every", type=int, default=500)
	parser.add_argument("--tb-logdir", type=str, default="runs/vdn_hockey_exp_1")
	parser.add_argument("--eval-episodes", type=int, default=3)
	parser.add_argument("--render-eval", action="store_true")
	parser.add_argument("--video-eval", action="store_true")
	parser.add_argument("--video-dir", type=str, default="videos/vdn_ice_hockey")
	parser.add_argument("--video-every", type=int, default=500)

	args = parser.parse_args()
	return TrainConfig(
		episodes=args.episodes,
		max_steps=args.max_steps,
		gamma=args.gamma,
		lr=args.lr,
		batch_size=args.batch_size,
		buffer_size=args.buffer_size,
		learning_starts=args.learning_starts,
		load_checkpoint=args.load_checkpoint,
		train_freq=args.train_freq,
		target_update_freq=args.target_update_freq,
		eps_start=args.eps_start,
		eps_end=args.eps_end,
		eps_decay_steps=args.eps_decay_steps,
		seed=args.seed,
		device=args.device,
		checkpoint_dir=args.checkpoint_dir,
		checkpoint_every=args.checkpoint_every,
		tensorboard_logdir=args.tb_logdir,
	), args.eval_episodes, args.render_eval, args.video_eval, args.video_dir, args.video_every


if __name__ == "__main__":
	train_cfg, eval_episodes, render_eval, video_eval, video_dir, video_every = parse_args()
	model, env_agents = train_vdn(train_cfg)
	evaluate(
		model,
		env_agents,
		episodes=eval_episodes,
		render=render_eval,
		record_video=video_eval,
		video_dir=video_dir,
		video_every=video_every,
		train_episode=train_cfg.episodes,
	)
