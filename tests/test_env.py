"""Unit tests for the StaticTapEnv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simulation import build_env_from_json
from simulation.env import (
    EnvConfig,
    RewardClippingWrapper,
    RewardNormalizationWrapper,
    StaticTapEnv,
)


def build_simple_config() -> EnvConfig:
    embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
    od_matrix = np.array([[0.0, 10.0], [0.0, 0.0]], dtype=np.float32)
    demands = np.array([10.0], dtype=np.float32)
    path_od_mapping = np.array([0, 0], dtype=np.int64)
    path_edge_incidence = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    freeflow = np.array([1.0, 2.0], dtype=np.float32)
    capacities = np.array([10.0, 5.0], dtype=np.float32)
    return EnvConfig(
        embeddings=embeddings,
        od_matrix=od_matrix,
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


def test_build_env_from_json_produces_functional_env():
    base_dir = Path(__file__).resolve().parent.parent
    scenario_dir = base_dir / "src" / "data" / "scenarios" / "test"
    network_path = scenario_dir / "test_network.json"
    demand_path = scenario_dir / "test_demand.json"

    scenario, path_set, env = build_env_from_json(network_path, demand_path, k_paths=2)

    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "od_demands" in info

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    _, reward, terminated, truncated, step_info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert isinstance(reward, float)
    assert "path_flows" in step_info

    flattened_embeddings = np.concatenate(
        [
            tensor.detach().cpu().numpy().reshape(-1)
            for _, tensor in sorted(scenario.embeddings.items())
        ]
    )
    np.testing.assert_allclose(
        env.embeddings.detach().cpu().numpy(), flattened_embeddings
    )
    np.testing.assert_allclose(
        env.od_matrix.detach().cpu().numpy(), scenario.od,
    )
    np.testing.assert_allclose(
        env.od_demands.detach().cpu().numpy(), path_set.od_demands.cpu().numpy()
    )
    np.testing.assert_allclose(
        env.path_od_mapping.detach().cpu().numpy(),
        path_set.path_od_mapping.cpu().numpy(),
    )
    np.testing.assert_allclose(
        env.path_edge_incidence.detach().cpu().numpy(),
        path_set.path_edge_incidence.cpu().numpy(),
    )
    np.testing.assert_allclose(
        env.freeflow_times.detach().cpu().numpy(),
        scenario.graph.freeflow_travel_time.detach().cpu().numpy(),
    )
    np.testing.assert_allclose(
        env.capacities.detach().cpu().numpy(),
        scenario.graph.capacity.detach().cpu().numpy(),
    )
