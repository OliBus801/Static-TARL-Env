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
        self.od_demands = torch.as_tensor(self.od_demands, dtype=torch.int32).view(-1)
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


class FlowConsistentMultiDiscrete(spaces.MultiDiscrete):
    """MultiDiscrete space that enforces OD demand consistency."""

    def __init__(
        self,
        nvec: np.ndarray,
        path_od_mapping: np.ndarray,
        od_demands: np.ndarray,
    ) -> None:
        super().__init__(nvec)
        self.path_od_mapping = np.asarray(path_od_mapping, dtype=np.int64).reshape(-1)
        self.od_demands = np.asarray(od_demands, dtype=np.int64).reshape(-1)
        if self.path_od_mapping.shape[0] != self.nvec.shape[0]:
            raise ValueError(
                "path_od_mapping must have the same size as the number of paths"
            )
        self.od_to_paths: list[np.ndarray] = []
        for od_idx in range(self.od_demands.size):
            indices = np.where(self.path_od_mapping == od_idx)[0]
            self.od_to_paths.append(indices)

    def sample(self) -> np.ndarray:
        sample = np.zeros_like(self.nvec, dtype=np.int64)
        for od_idx, path_indices in enumerate(self.od_to_paths):
            if path_indices.size == 0:
                continue
            demand = int(self.od_demands[od_idx])
            if demand == 0:
                continue
            probs = np.ones(path_indices.size, dtype=np.float64) / float(path_indices.size)
            allocation = self.np_random.multinomial(demand, probs)
            sample[path_indices] = allocation
        return sample

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x)
        if arr.dtype.kind not in {"i", "u"}:
            return False
        if not super().contains(arr):
            return False
        arr = arr.astype(np.int64, copy=False).reshape(-1)
        for od_idx, path_indices in enumerate(self.od_to_paths):
            if path_indices.size == 0:
                continue
            if int(np.sum(arr[path_indices])) != int(self.od_demands[od_idx]):
                return False
        return True


class StaticTapEnv(gym.Env):
    """Static TAP environment where an action is a set of path logits."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.device = device or torch.device("cpu")

        self.embeddings = config.embeddings.to(self.device, dtype=torch.float32)
        self.od_matrix = config.od_matrix.to(self.device, dtype=torch.float32)
        self.od_demands = config.od_demands.to(self.device, dtype=torch.int32)
        self.path_od_mapping = config.path_od_mapping.to(self.device)
        self.path_edge_incidence = config.path_edge_incidence.to(
            self.device, dtype=torch.float32
        )
        self.path_edge_incidence_int = self.path_edge_incidence.to(dtype=torch.int32)
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

        self.max_flow_per_path = torch.zeros(
            self.num_paths, dtype=torch.int32, device=self.device
        )
        for path_idx in range(self.num_paths):
            od_idx = int(self.path_od_mapping[path_idx].item())
            demand = int(self.od_demands[od_idx].item())
            self.max_flow_per_path[path_idx] = demand

        action_bounds = (
            self.max_flow_per_path.detach().cpu().numpy().astype(np.int64) + 1
        )
        path_od_mapping_np = self.path_od_mapping.detach().cpu().numpy().astype(np.int64)
        od_demands_np = self.od_demands.detach().cpu().numpy().astype(np.int64)
        self.action_space = FlowConsistentMultiDiscrete(
            action_bounds, path_od_mapping_np, od_demands_np
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

    def _validate_path_flows(self, action: np.ndarray | Tensor) -> Tensor:
        if isinstance(action, Tensor):
            action_array = action.detach().cpu().numpy()
        else:
            action_array = np.asarray(action)
        if action_array.dtype.kind not in {"i", "u"}:
            raise ValueError("Action must contain integer flow values.")

        flows = torch.as_tensor(action, dtype=torch.int32, device=self.device).view(-1)
        if flows.numel() != self.num_paths:
            raise ValueError("Action size does not match the number of available paths.")

        if torch.any(flows < 0):
            raise ValueError("Action cannot allocate negative flows.")

        if torch.any(flows > self.max_flow_per_path):
            raise ValueError("Action exceeds the available demand for at least one path.")

        for od_idx in range(self.num_od):
            mask = self.path_od_mapping == od_idx
            if not torch.any(mask):
                continue
            allocated = int(torch.sum(flows[mask]).item())
            demand = int(self.od_demands[od_idx].item())
            if allocated != demand:
                raise ValueError(
                    f"Action must allocate exactly {demand} agents for OD index {od_idx}."
                )

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

        path_flows_int = self._validate_path_flows(action)
        path_flows = path_flows_int.to(dtype=torch.float32)
        link_flows_int = torch.matmul(self.path_edge_incidence_int.T, path_flows_int)
        link_flows = link_flows_int.to(dtype=torch.float32)
        travel_times, system_cost = self._compute_link_costs(link_flows)

        reward = float(-system_cost.item())
        info = {
            "path_flows": path_flows_int.detach().cpu().numpy().astype(np.int32),
            "link_flows": link_flows_int.detach().cpu().numpy().astype(np.int32),
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
