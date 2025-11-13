"""PyTorch Geometric utilities for TAP scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency for configs
    yaml = None


def _parse_plaintext_params(filepath: Path) -> dict:
    params: Dict[str, float] = {}
    with open(filepath, "r", encoding="utf-8") as file:
        for raw_line in file:
            line, *_ = raw_line.split("#", maxsplit=1)
            if ":" not in line:
                continue
            key, value = line.split(":", maxsplit=1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            try:
                params[key] = float(value)
            except ValueError:
                continue
    return params


def load_params(filepath: Path) -> dict:
    if yaml is not None:
        with open(filepath, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    return _parse_plaintext_params(filepath)


CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "configs" / "params.yaml"
PARAMS = load_params(CONFIG_PATH)

ALPHA = PARAMS.get("alpha", 0.15)
BETA = PARAMS.get("beta", 4.0)


class TapScenario:
    """Utility class to manipulate TAP data stored in JSON files."""

    def __init__(self) -> None:
        self.graph: Data | None = None
        self.od: np.ndarray | None = None
        self.total_agents = 0
        self._node_id_to_idx: Dict[int, int] = {}

    def load_network_from_json(self, filepath: str | Path) -> Data:
        """Read a JSON network definition and build a PyG graph."""

        with open(filepath, "r", encoding="utf-8") as file:
            network_data = json.load(file)

        nodes: List[dict] = network_data.get("nodes", [])
        edges: List[dict] = network_data.get("edges", [])

        if not nodes:
            raise ValueError("The network JSON must contain at least one node.")

        if not edges:
            raise ValueError("The network JSON must contain at least one edge.")

        node_ids = []
        node_coordinates = []
        for node in nodes:
            if "id" not in node:
                raise ValueError("Each node entry must define an 'id'.")
            node_ids.append(int(node["id"]))
            node_coordinates.append(
                [float(node.get("x", 0.0)), float(node.get("y", 0.0))]
            )

        self._node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        edge_sources = []
        edge_targets = []
        capacities = []
        freeflows = []
        lengths = []
        for edge in edges:
            if "source" not in edge or "target" not in edge:
                raise ValueError("Each edge entry must define 'source' and 'target'.")
            try:
                origin = self._node_id_to_idx[int(edge["source"])]
                destination = self._node_id_to_idx[int(edge["target"])]
            except KeyError as exc:
                raise ValueError("Edge references unknown node ids.") from exc

            capacity = float(edge.get("capacity", 0.0))
            freeflow = float(edge.get("freeflow_travel_time", 0.0))

            origin_coord = node_coordinates[origin]
            destination_coord = node_coordinates[destination]
            length = float(
                np.linalg.norm(
                    np.array(destination_coord) - np.array(origin_coord)
                )
            )

            edge_sources.append(origin)
            edge_targets.append(destination)
            capacities.append(capacity)
            freeflows.append(freeflow)
            lengths.append(length)

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        x = torch.tensor(node_coordinates, dtype=torch.float32)
        capacity_tensor = torch.tensor(capacities, dtype=torch.float32)
        freeflow_tensor = torch.tensor(freeflows, dtype=torch.float32)
        length_tensor = torch.tensor(lengths, dtype=torch.float32)
        edge_attr = torch.stack(
            [capacity_tensor, freeflow_tensor, length_tensor], dim=1
        )

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.capacity = capacity_tensor
        data.freeflow_travel_time = freeflow_tensor
        data.length = length_tensor
        data.node_ids = torch.tensor(node_ids, dtype=torch.long)
        data.edge_ids = torch.arange(edge_index.size(1), dtype=torch.long)

        self.graph = data

        print(
            "✅ Successfully loaded network into PyTorch Geometric format!",
            f"🌐 Network Info: {data.num_nodes} nodes, {data.num_edges} edges.",
        )

        return data

    def load_demand_from_json(self, filepath: str | Path) -> np.ndarray:
        """Load OD matrix information from a JSON file."""

        with open(filepath, "r", encoding="utf-8") as file:
            demand_data = json.load(file)
        self.od = np.array(demand_data["matrix"], dtype=float)
        self.total_agents = int(np.sum(self.od))
        print(
            "✅ Successfully loaded demand from JSON file!",
            f"👥 Demand Info: Total agents = {self.total_agents}.",
        )
        return self.od

    def check_flow_consistency(self, assignment: List[float] | np.ndarray) -> bool:
        """Verify whether the assignment complies with flow conservation."""

        if self.graph is None or self.od is None:
            raise ValueError("Network and demand must be loaded before checking flows.")

        assignment_array = np.asarray(assignment, dtype=float)
        if assignment_array.size != self.graph.num_edges:
            raise ValueError(
                "Assignment size must match the number of edges in the graph."
            )

        flows = np.zeros(self.graph.num_nodes, dtype=float)
        flows += np.sum(self.od, axis=1)
        flows -= np.sum(self.od, axis=0)

        edge_index = self.graph.edge_index.cpu().numpy()
        for edge_idx in range(edge_index.shape[1]):
            origin = int(edge_index[0, edge_idx])
            destination = int(edge_index[1, edge_idx])
            flow_value = assignment_array[edge_idx]
            flows[origin] -= flow_value
            flows[destination] += flow_value

        if not np.allclose(flows, 0.0):
            inconsistent_nodes = np.where(~np.isclose(flows, 0.0))[0]
            print(
                "❌ Flow consistency check failed for nodes:",
                inconsistent_nodes,
                "with flow values:",
                flows[inconsistent_nodes],
            )
            return False
        print("✅ Flow consistency check passed.")
        return True

    def calculate_system_cost(self, assignment: List[float] | np.ndarray) -> float:
        """Compute the total system travel time using the BPR cost function."""

        if self.graph is None:
            raise ValueError("Network must be loaded before calculating costs.")

        assignment_array = np.asarray(assignment, dtype=float)
        if assignment_array.size != self.graph.num_edges:
            raise ValueError(
                "Assignment size must match the number of edges in the graph."
            )

        if not self.check_flow_consistency(assignment_array):
            raise ValueError("The provided assignment does not verify flow consistency.")

        capacities = self.graph.capacity.cpu().numpy()
        freeflows = self.graph.freeflow_travel_time.cpu().numpy()
        total_cost = 0.0
        for idx in range(self.graph.num_edges):
            travel_time = freeflows[idx] * (
                1 + ALPHA * (assignment_array[idx] / capacities[idx]) ** BETA
            )
            total_cost += travel_time * assignment_array[idx]

        print(f"Coût total du système : {total_cost}")
        return float(total_cost)
