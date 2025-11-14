"""Gymnasium environment for the static traffic assignment problem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import Tensor


@dataclass
class EnvConfig:
    """Utility container with the tensors required to build the environment."""

    embeddings: Tensor | np.ndarray
    od_matrix: Tensor | np.ndarray
    od_demands: Tensor | np.ndarray
    path_od_mapping: Tensor | np.ndarray
    path_edge_incidence: Tensor | np.ndarray
    freeflow_times: Tensor | np.ndarray
    capacities: Tensor | np.ndarray
    alpha: float = 0.15
    beta: float = 4.0

    def __post_init__(self) -> None:
        """Convert inputs to tensors to ease interoperability with numpy."""

        self.embeddings = torch.as_tensor(self.embeddings, dtype=torch.float32)
        self.od_matrix = torch.as_tensor(self.od_matrix, dtype=torch.float32)
        self.od_demands = torch.as_tensor(self.od_demands, dtype=torch.float32).view(-1)
        self.path_od_mapping = torch.as_tensor(
            self.path_od_mapping, dtype=torch.long
        ).view(-1)
        self.path_edge_incidence = torch.as_tensor(
            self.path_edge_incidence, dtype=torch.float32
        )
        self.freeflow_times = torch.as_tensor(
            self.freeflow_times, dtype=torch.float32
        ).view(-1)
        self.capacities = torch.as_tensor(self.capacities, dtype=torch.float32).view(-1)


class StaticTapEnv(gym.Env):
    """Static TAP environment where an action is a set of path logits."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")

        self.embeddings = config.embeddings.to(self.device, dtype=torch.float32)
        self.od_matrix = config.od_matrix.to(self.device, dtype=torch.float32)
        self.od_demands = config.od_demands.to(self.device, dtype=torch.float32)
        self.path_od_mapping = config.path_od_mapping.to(self.device)
        self.path_edge_incidence = config.path_edge_incidence.to(
            self.device, dtype=torch.float32
        )
        self.freeflow_times = config.freeflow_times.to(self.device, dtype=torch.float32)
        self.capacities = config.capacities.to(self.device, dtype=torch.float32)
        self.alpha = config.alpha
        self.beta = config.beta

        if self.od_demands.ndim != 1:
            raise ValueError("od_demands must be a 1-D tensor with per-OD demand values.")

        if self.path_od_mapping.ndim != 1:
            raise ValueError("path_od_mapping must be 1-D with the OD index of each path.")

        if self.path_edge_incidence.ndim != 2:
            raise ValueError("path_edge_incidence must be 2-D (paths x edges).")

        self.num_paths = int(self.path_edge_incidence.size(0))
        self.num_edges = int(self.path_edge_incidence.size(1))
        self.num_od = int(self.od_demands.numel())

        if self.path_od_mapping.numel() != self.num_paths:
            raise ValueError("path_od_mapping must have the same length as the number of paths.")

        obs_dim = int(self.embeddings.numel() + self.od_matrix.numel())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_paths,), dtype=np.float32
        )

    def _build_observation(self) -> np.ndarray:
        obs = torch.cat(
            [self.embeddings.flatten(), self.od_matrix.flatten()], dim=0
        ).to(torch.float32)
        return obs.detach().cpu().numpy()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        obs = self._build_observation()
        info = {"od_demands": self.od_demands.detach().cpu().numpy()}
        return obs, info

    def _logits_to_path_flows(self, action: np.ndarray | Tensor) -> Tensor:
        logits = torch.as_tensor(action, dtype=torch.float32, device=self.device).view(-1)
        if logits.numel() != self.num_paths:
            raise ValueError("Action size does not match the number of available paths.")

        flows = torch.zeros_like(logits)
        for od_idx in range(self.num_od):
            mask = self.path_od_mapping == od_idx
            if not torch.any(mask):
                continue
            od_logits = logits[mask]
            probs = torch.softmax(od_logits, dim=0)
            flows[mask] = probs * self.od_demands[od_idx]
        return flows

    def _compute_link_costs(self, link_flows: Tensor) -> tuple[Tensor, Tensor]:
        ratios = torch.zeros_like(link_flows)
        positive_capacity = self.capacities > 0
        ratios[positive_capacity] = (
            link_flows[positive_capacity] / self.capacities[positive_capacity]
        )
        travel_times = self.freeflow_times * (1 + self.alpha * torch.pow(ratios, self.beta))
        system_cost = torch.sum(travel_times * link_flows)
        return travel_times, system_cost

    def step(self, action: np.ndarray):
        if not self.action_space.contains(action):
            raise ValueError("Action is outside of the defined action space.")

        path_flows = self._logits_to_path_flows(action)
        link_flows = torch.matmul(self.path_edge_incidence.T, path_flows)
        travel_times, system_cost = self._compute_link_costs(link_flows)

        reward = float(-system_cost.item())
        info = {
            "path_flows": path_flows.detach().cpu().numpy(),
            "link_flows": link_flows.detach().cpu().numpy(),
            "travel_times": travel_times.detach().cpu().numpy(),
        }
        obs = self._build_observation()
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, info


class RewardNormalizationWrapper(gym.RewardWrapper):
    """Normalize rewards online using running statistics."""

    def __init__(self, env: gym.Env, epsilon: float = 1e-8) -> None:
        super().__init__(env)
        self.count = epsilon
        self.mean = 0.0
        self.m2 = 0.0

    def reward(self, reward: float) -> float:
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.m2 += delta * delta2
        variance = self.m2 / max(self.count - 1, 1.0)
        std = float(np.sqrt(max(variance, 0.0)) + 1e-8)
        return reward / std


class RewardClippingWrapper(gym.RewardWrapper):
    """Clip rewards to keep them within a bounded range."""

    def __init__(self, env: gym.Env, min_reward: float, max_reward: float) -> None:
        super().__init__(env)
        if min_reward > max_reward:
            raise ValueError("min_reward must be less or equal than max_reward.")
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward: float) -> float:
        return float(np.clip(reward, self.min_reward, self.max_reward))
