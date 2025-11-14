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
    demands = np.array([10], dtype=np.int32)
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


def build_feasible_action(environment: StaticTapEnv) -> np.ndarray:
    action = np.zeros(environment.num_paths, dtype=np.int32)
    mapping = environment.path_od_mapping.detach().cpu().numpy()
    od_demands = environment.od_demands.detach().cpu().numpy()
    for od_idx in range(environment.num_od):
        indices = np.where(mapping == od_idx)[0]
        if indices.size == 0:
            continue
        action[indices[0]] = int(od_demands[od_idx])
    return action


def test_action_allocation_matches_demand(env):
    env.reset()
    action = np.array([5, 5], dtype=np.int32)
    _, _, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert info["path_flows"].dtype == np.int32
    assert np.sum(info["path_flows"]) == 10
    assert info["link_flows"].dtype == np.int32
    assert info["link_flows"][0] == 5
    assert info["link_flows"][1] == 5


def test_reward_prefers_fast_path(env):
    env.reset()
    faster_action = np.array([10, 0], dtype=np.int32)
    _, reward_fast, _, _, _ = env.step(faster_action)

    env.reset()
    slower_action = np.array([0, 10], dtype=np.int32)
    _, reward_slow, _, _, _ = env.step(slower_action)

    assert reward_fast > reward_slow


def test_reward_wrappers_clip_and_normalize(env):
    wrapped_env = RewardClippingWrapper(
        RewardNormalizationWrapper(env), min_reward=-1.0, max_reward=1.0
    )
    wrapped_env.reset()
    action = np.array([10, 0], dtype=np.int32)
    _, reward, _, _, _ = wrapped_env.step(action)

    assert -1.0 <= reward <= 1.0


def test_action_requires_integer_and_complete_flows(env):
    env.reset()
    float_action = np.array([5.0, 5.0], dtype=np.float32)
    with pytest.raises(ValueError):
        env.step(float_action)

    env.reset()
    incomplete_action = np.array([6, 3], dtype=np.int32)
    with pytest.raises(ValueError):
        env.step(incomplete_action)


def test_action_space_sampling_respects_flow_consistency(env):
    env.reset()
    od_demands = env.od_demands.detach().cpu().numpy()
    mapping = env.path_od_mapping.detach().cpu().numpy()
    invalid = np.array([5.0, 5.0], dtype=np.float32)
    assert not env.action_space.contains(invalid)
    for _ in range(25):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        for od_idx in range(env.num_od):
            indices = np.where(mapping == od_idx)[0]
            if indices.size == 0:
                continue
            assert np.sum(action[indices]) == od_demands[od_idx]


def test_build_env_from_json_produces_functional_env():
    base_dir = Path(__file__).resolve().parent.parent
    scenario_dir = base_dir / "src" / "data" / "scenarios" / "test"
    network_path = scenario_dir / "test_network.json"
    demand_path = scenario_dir / "test_demand.json"

    scenario, path_set, env = build_env_from_json(network_path, demand_path, k_paths=2)

    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "od_demands" in info

    action = build_feasible_action(env)
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
    np.testing.assert_array_equal(
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
