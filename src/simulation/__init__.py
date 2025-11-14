"""Static TARL Simulator - Traffic Assignment Problem Implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .env import EnvConfig, StaticTapEnv
from .network import ALPHA, BETA, TapScenario
from .paths import PathSet, build_path_set

__all__ = [
    "EnvConfig",
    "PathSet",
    "StaticTapEnv",
    "TapScenario",
    "build_env_from_json",
    "build_path_set",
]

__version__ = "0.1.0"


def _flatten_embeddings(embeddings: Dict[str, Tensor]) -> Tensor:
    """Concatenate all available scenario embeddings into a single tensor."""

    if not embeddings:
        return torch.empty(0, dtype=torch.float32)
    flattened: List[Tensor] = []
    for key in sorted(embeddings.keys()):
        tensor = torch.as_tensor(embeddings[key], dtype=torch.float32)
        flattened.append(tensor.reshape(-1))
    return torch.cat(flattened, dim=0)


def build_env_from_json(
    network_path: str | Path,
    demand_path: str | Path,
    *,
    k_paths: int = 3,
    device: torch.device | None = None,
) -> Tuple[TapScenario, PathSet, StaticTapEnv]:
    """Instantiate a StaticTapEnv directly from JSON network and demand files."""

    scenario = TapScenario()
    graph = scenario.load_network_from_json(network_path)
    od_matrix = scenario.load_demand_from_json(demand_path)

    path_set = build_path_set(graph, od_matrix, k=k_paths)
    embeddings = _flatten_embeddings(scenario.compute_embeddings())
    od_matrix_tensor = torch.as_tensor(od_matrix, dtype=torch.float32)

    freeflow_times = getattr(graph, "freeflow_travel_time", None)
    capacities = getattr(graph, "capacity", None)
    if freeflow_times is None or capacities is None:
        raise ValueError("Graph must contain freeflow_travel_time and capacity attributes.")

    config = EnvConfig(
        embeddings=embeddings,
        od_matrix=od_matrix_tensor,
        od_demands=torch.as_tensor(path_set.od_demands, dtype=torch.float32),
        path_od_mapping=torch.as_tensor(path_set.path_od_mapping, dtype=torch.long),
        path_edge_incidence=torch.as_tensor(path_set.path_edge_incidence, dtype=torch.float32),
        freeflow_times=torch.as_tensor(freeflow_times, dtype=torch.float32).flatten(),
        capacities=torch.as_tensor(capacities, dtype=torch.float32).flatten(),
        alpha=ALPHA,
        beta=BETA,
    )

    env = StaticTapEnv(config, device=device)
    return scenario, path_set, env
