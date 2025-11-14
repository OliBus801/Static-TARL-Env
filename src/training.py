"""Simple training utilities for Static-TARL environments."""
from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import yaml
from gymnasium import spaces
from tqdm import tqdm

from .rl_models.q_learning_agent import QLearningAgent
from .simulation import StaticTapEnv, build_env_from_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "src" / "data" / "configs" / "params.yaml"


def _load_params(params_path: Path) -> Dict[str, Any]:
    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_required_path(params_path: Path, candidate: str) -> Path:
    """Resolve ``candidate`` relative to useful anchors and validate its existence."""

    path = Path(candidate)
    if path.is_absolute():
        resolved = path
    else:
        params_dir = params_path.parent
        guesses = [params_dir, PROJECT_ROOT, Path.cwd()]
        resolved = (params_dir / path).resolve()
        for base in guesses:
            tentative = (base / path).resolve()
            if tentative.exists():
                resolved = tentative
                break
    if not resolved.exists():
        raise FileNotFoundError(f"Unable to resolve path '{candidate}' relative to the configuration file.")
    return resolved


def _build_environment(params_path: Path, scenario_cfg: Dict[str, Any]) -> StaticTapEnv:
    network_path = _resolve_required_path(
        params_path, scenario_cfg.get("network_path", "src/data/scenarios/test/test_network.json")
    )
    demand_path = _resolve_required_path(
        params_path, scenario_cfg.get("demand_path", "src/data/scenarios/test/test_demand.json")
    )
    k_paths = int(scenario_cfg.get("k_paths", 3))

    _, _, env = build_env_from_json(network_path, demand_path, k_paths=k_paths)
    return env


def _build_action_library(
    env: StaticTapEnv,
    pool_size: int,
    seed: int | None,
) -> List[np.ndarray]:
    if pool_size <= 0:
        raise ValueError("action_library_size must be strictly positive.")

    if seed is not None:
        env.action_space.seed(seed)

    samples: List[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    max_attempts = pool_size * 20
    while len(samples) < pool_size and attempts < max_attempts:
        attempts += 1
        action = env.action_space.sample()
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        key = tuple(int(x) for x in action_arr)
        if key in seen:
            continue
        seen.add(key)
        samples.append(action_arr.copy())

    if len(samples) < pool_size:
        raise RuntimeError(
            "Unable to build an action library with unique samples from the environment's action space."
        )
    return samples


def _state_to_key(observation: np.ndarray | Sequence[float]) -> tuple[float, ...]:
    obs = np.asarray(observation, dtype=np.float32).flatten()
    return tuple(np.round(obs, decimals=6))


def _instantiate_agent(agent_type: str, action_space: spaces.Discrete, agent_cfg: Dict[str, Any]) -> QLearningAgent:
    if agent_type != "q_learning":
        raise ValueError(f"Unsupported agent_type '{agent_type}'. Only 'q_learning' is available.")

    return QLearningAgent(
        action_space=action_space,
        learning_rate=float(agent_cfg.get("learning_rate", 0.1)),
        discount=float(agent_cfg.get("discount", 0.99)),
        initial_epsilon=float(agent_cfg.get("initial_epsilon", 1.0)),
        final_epsilon=float(agent_cfg.get("final_epsilon", 0.05)),
        epsilon_decay_rate=float(agent_cfg.get("epsilon_decay_rate", 0.995)),
    )


def run_training(agent_type: str, params_path: str = str(DEFAULT_PARAMS_PATH)) -> Dict[str, Any]:
    """Run a lightweight training loop and return aggregated metrics."""

    config_path = Path(params_path).resolve()
    params = _load_params(config_path)
    training_cfg = params.get("training", {})
    num_episodes = int(training_cfg.get("num_episodes", 1))
    rolling_window = int(training_cfg.get("rolling_window", max(1, min(10, num_episodes))))
    action_library_size = int(training_cfg.get("action_library_size", 1))
    agent_cfg = training_cfg.get("agent", {})
    seeds: Iterable[int | None] = training_cfg.get("seeds", params.get("seeds", [None]))

    base_seed = None
    seeds_list = list(seeds)
    if seeds_list:
        base_seed = seeds_list[0]

    if base_seed is not None:
        random_seed = int(base_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    env = _build_environment(config_path, training_cfg.get("scenario", {}))
    action_library = _build_action_library(env, action_library_size, base_seed)
    discrete_space = spaces.Discrete(len(action_library))
    agent = _instantiate_agent(agent_type, discrete_space, agent_cfg)

    episode_returns: List[float] = []
    window = deque(maxlen=max(1, rolling_window))
    progress = tqdm(range(num_episodes), desc="Training")

    seed_cycle = seeds_list if seeds_list else [None]
    for episode_idx in progress:
        episode_seed = seed_cycle[episode_idx % len(seed_cycle)]
        obs, _ = env.reset(seed=None if episode_seed is None else int(episode_seed))
        state = _state_to_key(obs)
        done = False
        total_reward = 0.0

        while not done:
            action_idx = agent.get_action(state)
            env_action = action_library[action_idx]
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            total_reward += reward
            next_state = _state_to_key(next_obs)
            done = bool(terminated or truncated)
            agent.update(state, action_idx, reward, next_state, done)
            state = next_state

        agent.epsilon_decay()
        episode_returns.append(float(total_reward))
        window.append(total_reward)
        progress.set_postfix(mean_return=float(np.mean(list(window))))

    env.close()
    progress.close()

    return {
        "returns": episode_returns,
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "num_episodes": num_episodes,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train agents inside the Static-TARL environment.")
    parser.add_argument(
        "--agent",
        default="q_learning",
        choices=["q_learning"],
        help="Type of agent to train.",
    )
    parser.add_argument(
        "--params",
        default=str(DEFAULT_PARAMS_PATH),
        help="Path to the YAML configuration file containing hyperparameters.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    stats = run_training(args.agent, args.params)
    print(
        "Training finished:",
        f"episodes={stats['num_episodes']}",
        f"mean_return={stats['mean_return']:.2f}",
    )


if __name__ == "__main__":
    main()
