"""Unit tests for the StaticTapEnv."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from simulation.env import (
    EnvConfig,
    RewardClippingWrapper,
    RewardNormalizationWrapper,
    StaticTapEnv,
)


def build_simple_config() -> EnvConfig:
    embeddings = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    od_matrix = torch.tensor([[0.0, 10.0], [0.0, 0.0]], dtype=torch.float32)
    od_features = od_matrix
    demands = torch.tensor([10.0], dtype=torch.float32)
    path_od_mapping = torch.tensor([0, 0], dtype=torch.long)
    path_edge_incidence = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    freeflow = torch.tensor([1.0, 2.0], dtype=torch.float32)
    capacities = torch.tensor([10.0, 5.0], dtype=torch.float32)
    return EnvConfig(
        embeddings=embeddings,
        od_features=od_features,
        od_demands=demands,
        path_od_mapping=path_od_mapping,
        path_edge_incidence=path_edge_incidence,
        freeflow_times=freeflow,
        capacities=capacities,
    )


@pytest.fixture()
def env():
    return StaticTapEnv(build_simple_config())


def test_action_allocation_matches_demand(env):
    env.reset()
    action = np.array([0.0, 0.0], dtype=np.float32)
    _, _, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert np.isclose(np.sum(info["path_flows"]), 10.0)
    assert np.isclose(info["link_flows"][0], 5.0)
    assert np.isclose(info["link_flows"][1], 5.0)


def test_reward_prefers_fast_path(env):
    env.reset()
    faster_action = np.array([5.0, -5.0], dtype=np.float32)
    _, reward_fast, _, _, _ = env.step(faster_action)

    env.reset()
    slower_action = np.array([-5.0, 5.0], dtype=np.float32)
    _, reward_slow, _, _, _ = env.step(slower_action)

    assert reward_fast > reward_slow


def test_reward_wrappers_clip_and_normalize(env):
    wrapped_env = RewardClippingWrapper(
        RewardNormalizationWrapper(env), min_reward=-1.0, max_reward=1.0
    )
    wrapped_env.reset()
    action = np.array([5.0, -5.0], dtype=np.float32)
    _, reward, _, _, _ = wrapped_env.step(action)

    assert -1.0 <= reward <= 1.0
