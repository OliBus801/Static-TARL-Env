"""PyTorch Geometric utilities for TAP scenarios."""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data
import yaml

from .gnn_encoder import GNNEncoder, GNNEncoderConfig


# Load configuration parameters --------------
def load_params(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

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
        self.encoder: GNNEncoder | None = None
        self.embeddings: Dict[str, torch.Tensor] = {}

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
            # Here we compute Euclidean distance as length. TODO: Decide if we prefer something else.
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
        # Creating node feature matrix (only coordinates as of now)    
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

    def build_gnn_encoder(self, **kwargs) -> GNNEncoder:
        """Create a default GNN encoder based on the loaded graph."""

        if self.graph is None:
            raise ValueError("Load a graph before building an encoder.")

        config = GNNEncoderConfig(
            node_in_channels=int(self.graph.x.size(1)),
            edge_in_channels=int(self.graph.edge_attr.size(1)),
            **kwargs,
        )
        self.encoder = GNNEncoder(config)
        return self.encoder

    def compute_embeddings(self, encoder: GNNEncoder | None = None) -> Dict[str, torch.Tensor]:
        """Generate and expose embeddings for reuse in the scenario."""

        if self.graph is None:
            raise ValueError("Graph must be loaded before computing embeddings.")

        if encoder is not None:
            self.encoder = encoder
        elif self.encoder is None:
            self.build_gnn_encoder()

        assert self.encoder is not None
        self.embeddings = self.encoder(self.graph)
        return self.embeddings

    def load_demand_from_json(self, filepath: str | Path) -> np.ndarray:
        """Load OD matrix information from a JSON file."""

        with open(filepath, "r", encoding="utf-8") as file:
            demand_data = json.load(file)

        raw_matrix = np.array(demand_data["matrix"], dtype=float)
        if np.any(raw_matrix < 0):
            raise ValueError("Demand matrix must contain non-negative values.")

        rounded = np.rint(raw_matrix)
        if not np.allclose(raw_matrix, rounded):
            raise ValueError(
                "Demand matrix must contain integer-compatible demand counts."
            )

        self.od = rounded.astype(np.int32)
        self.total_agents = int(np.sum(self.od))
        print(
            "✅ Successfully loaded demand from JSON file!",
            f"👥 Demand Info: Total agents = {self.total_agents}.",
        )
        return self.od

    def check_flow_consistency(self, assignment: np.ndarray) -> bool:
        """Verify whether the assignment complies with flow conservation."""

        assignment_array = np.asarray(assignment)
        if assignment_array.dtype.kind not in {"i", "u"}:
            raise TypeError("Flow assignments must contain integer values.")

        assignment_int = assignment_array.astype(np.int64, copy=False)

        # supply = outflow demand - inflow demand per node
        supply = np.sum(self.od, axis=1) - np.sum(self.od, axis=0)

        edge_index = self.graph.edge_index.cpu().numpy().astype(np.int64)
        origins = edge_index[0]
        destinations = edge_index[1]
        n_nodes = int(self.graph.num_nodes)

        # vectorised aggregation of flows per node
        outflows = np.bincount(origins, weights=assignment_int, minlength=n_nodes)
        inflows = np.bincount(destinations, weights=assignment_int, minlength=n_nodes)

        flows = supply - outflows + inflows

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

    def calculate_system_cost(self, assignment: np.ndarray) -> float:
        """Compute the total system travel time using the BPR cost function."""

        if self.graph is None:
            raise ValueError("Network must be loaded before calculating costs.")

        assignment_array = np.asarray(assignment)
        if assignment_array.dtype.kind not in {"i", "u"}:
            raise TypeError("Assignment must be an array of integer flows.")

        if assignment_array.size != self.graph.num_edges:
            raise ValueError(
                "Assignment size must match the number of edges in the graph."
            )

        if not self.check_flow_consistency(assignment_array):
            raise ValueError("The provided assignment does not verify flow consistency.")

        flows = assignment_array.astype(float, copy=False)
        capacities = self.graph.capacity.cpu().numpy()
        freeflows = self.graph.freeflow_travel_time.cpu().numpy()

        ratio = np.zeros_like(flows, dtype=float)
        nonzero_mask = capacities > 0
        ratio[nonzero_mask] = flows[nonzero_mask] / capacities[nonzero_mask]

        travel_times = freeflows * (1 + ALPHA * ratio**BETA)
        total_cost = float(np.sum(travel_times * flows))

        print(f"Coût total du système : {total_cost}")
        return float(total_cost)
