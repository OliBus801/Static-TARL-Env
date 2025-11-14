"""Path generation utilities operating directly on PyG graphs."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


@dataclass
class PathSet:
    """Container gathering all tensors derived from the k-shortest paths."""

    od_pairs: List[Tuple[int, int]]
    od_demands: Tensor
    path_od_mapping: torch.LongTensor
    path_edge_incidence: Tensor
    path_costs: Tensor
    paths: List[List[int]]

    @property
    def num_paths(self) -> int:
        return int(self.path_costs.numel())


@dataclass(order=True)
class _Path:
    cost: float
    nodes: List[int]
    edges: List[int]


def _edge_lengths_from_graph(graph: Data) -> Tensor:
    if hasattr(graph, "length") and graph.length is not None:
        return torch.as_tensor(graph.length, dtype=torch.float64).flatten()
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        # Assume the last column contains lengths if the attribute exists.
        attr = torch.as_tensor(graph.edge_attr, dtype=torch.float64)
        if attr.ndim == 2:
            return attr[:, -1].flatten()
    num_edges = int(graph.edge_index.size(1))
    return torch.ones(num_edges, dtype=torch.float64)


def _build_adjacency(graph: Data) -> List[List[Tuple[int, int]]]:
    edge_index = graph.edge_index.cpu()
    num_edges = edge_index.size(1)
    num_nodes = int(graph.num_nodes)
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for edge_id in range(num_edges):
        origin = int(edge_index[0, edge_id])
        destination = int(edge_index[1, edge_id])
        adjacency[origin].append((destination, edge_id))
    return adjacency


def _shortest_path(
    adjacency: Sequence[Sequence[Tuple[int, int]]],
    edge_lengths: Sequence[float],
    source: int,
    target: int,
    forbidden_edges: Iterable[int] | None = None,
    forbidden_nodes: Iterable[int] | None = None,
) -> _Path | None:
    if source == target:
        return _Path(cost=0.0, nodes=[source], edges=[])

    num_nodes = len(adjacency)
    forbidden_edges = set(forbidden_edges or [])
    forbidden_nodes = set(forbidden_nodes or [])

    distances = [float("inf")] * num_nodes
    predecessors = [-1] * num_nodes
    predecessor_edge = [-1] * num_nodes
    distances[source] = 0.0

    heap: List[Tuple[float, int]] = [(0.0, source)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node == target:
            break
        if cost > distances[node]:
            continue
        for neighbor, edge_idx in adjacency[node]:
            if edge_idx in forbidden_edges:
                continue
            if neighbor != target and neighbor in forbidden_nodes:
                continue
            new_cost = cost + float(edge_lengths[edge_idx])
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                predecessors[neighbor] = node
                predecessor_edge[neighbor] = edge_idx
                heapq.heappush(heap, (new_cost, neighbor))

    if distances[target] == float("inf"):
        return None

    nodes = [target]
    edges: List[int] = []
    cursor = target
    while cursor != source:
        edge_idx = predecessor_edge[cursor]
        if edge_idx == -1:
            return None
        edges.append(edge_idx)
        cursor = predecessors[cursor]
        nodes.append(cursor)
    nodes.reverse()
    edges.reverse()
    return _Path(cost=distances[target], nodes=nodes, edges=edges)


def _yen_k_shortest_paths(
    adjacency: Sequence[Sequence[Tuple[int, int]]],
    edge_lengths: Sequence[float],
    source: int,
    target: int,
    k: int,
) -> List[_Path]:
    if k <= 0:
        return []
    first_path = _shortest_path(adjacency, edge_lengths, source, target)
    if first_path is None:
        return []

    shortest_paths = [first_path]
    candidates: List[Tuple[float, int, _Path]] = []
    counter = count()

    for _ in range(1, k):
        last_path = shortest_paths[-1]
        for spur_index in range(len(last_path.nodes) - 1):
            spur_node = last_path.nodes[spur_index]
            root_nodes = last_path.nodes[: spur_index + 1]
            root_edges = last_path.edges[:spur_index]

            removed_edges = set()
            for path in shortest_paths:
                if path.nodes[: spur_index + 1] == root_nodes and len(path.edges) > spur_index:
                    removed_edges.add(path.edges[spur_index])

            forbidden_nodes = set(root_nodes[:-1])
            spur_path = _shortest_path(
                adjacency,
                edge_lengths,
                spur_node,
                target,
                forbidden_edges=removed_edges,
                forbidden_nodes=forbidden_nodes,
            )
            if spur_path is None or not spur_path.edges:
                continue

            total_nodes = root_nodes[:-1] + spur_path.nodes
            total_edges = root_edges + spur_path.edges
            root_cost = sum(float(edge_lengths[idx]) for idx in root_edges)
            total_cost = root_cost + spur_path.cost
            candidate = _Path(cost=total_cost, nodes=total_nodes, edges=total_edges)
            candidate_signature = tuple(candidate.edges)
            existing_signatures = {tuple(path.edges) for path in shortest_paths}
            if candidate_signature in existing_signatures:
                continue
            heapq.heappush(
                candidates,
                (total_cost, next(counter), candidate),
            )

        if not candidates:
            break
        _, _, next_path = heapq.heappop(candidates)
        shortest_paths.append(next_path)

    return shortest_paths


def _extract_od_pairs(od_matrix: Tensor | np.ndarray) -> Tuple[List[Tuple[int, int]], Tensor]:
    od_tensor = torch.as_tensor(od_matrix, dtype=torch.float32)
    if od_tensor.ndim != 2 or od_tensor.size(0) != od_tensor.size(1):
        raise ValueError("od_matrix must be a square matrix with origin/destination demands.")
    od_pairs: List[Tuple[int, int]] = []
    od_demands: List[float] = []
    for origin in range(od_tensor.size(0)):
        for destination in range(od_tensor.size(1)):
            if origin == destination:
                continue
            demand = float(od_tensor[origin, destination].item())
            if demand <= 0:
                continue
            od_pairs.append((origin, destination))
            od_demands.append(demand)
    if not od_pairs:
        return [], torch.empty(0, dtype=torch.float32)
    return od_pairs, torch.tensor(od_demands, dtype=torch.float32)


def build_path_set(graph: Data, od_matrix: Tensor | np.ndarray, k: int = 3) -> PathSet:
    """Compute the path-edge incidence tensors for the k-shortest paths per OD."""

    if graph.edge_index is None:
        raise ValueError("Graph must contain edge_index information to compute paths.")
    if graph.num_nodes is None:
        raise ValueError("Graph must define num_nodes to compute paths.")

    adjacency = _build_adjacency(graph)
    edge_lengths = _edge_lengths_from_graph(graph)
    od_pairs, od_demands = _extract_od_pairs(od_matrix)

    num_edges = int(graph.edge_index.size(1))
    if not od_pairs:
        return PathSet(
            od_pairs=[],
            od_demands=od_demands,
            path_od_mapping=torch.empty(0, dtype=torch.long),
            path_edge_incidence=torch.zeros((0, num_edges), dtype=torch.float32),
            path_costs=torch.empty(0, dtype=torch.float32),
            paths=[],
        )

    path_rows: List[Tensor] = []
    path_costs: List[float] = []
    path_od_indices: List[int] = []
    path_nodes: List[List[int]] = []

    for od_idx, (origin, destination) in enumerate(od_pairs):
        k_paths = _yen_k_shortest_paths(adjacency, edge_lengths, origin, destination, k)
        for path in k_paths:
            incidence_row = torch.zeros(num_edges, dtype=torch.float32)
            for edge_idx in path.edges:
                incidence_row[edge_idx] += 1.0
            path_rows.append(incidence_row)
            path_costs.append(float(path.cost))
            path_od_indices.append(od_idx)
            path_nodes.append(path.nodes)

    if not path_rows:
        incidence = torch.zeros((0, num_edges), dtype=torch.float32)
        costs = torch.empty(0, dtype=torch.float32)
        mapping = torch.empty(0, dtype=torch.long)
    else:
        incidence = torch.stack(path_rows, dim=0)
        costs = torch.tensor(path_costs, dtype=torch.float32)
        mapping = torch.tensor(path_od_indices, dtype=torch.long)

    return PathSet(
        od_pairs=od_pairs,
        od_demands=od_demands,
        path_od_mapping=mapping,
        path_edge_incidence=incidence,
        path_costs=costs,
        paths=path_nodes,
    )


def all_or_nothing_assignment(
    path_set: PathSet,
    od_demands: Tensor | np.ndarray | None = None,
) -> Tuple[Tensor, Tensor]:
    """Deterministic AoN assignment placing each OD demand on its cheapest path."""

    if path_set.num_paths == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty

    if od_demands is None:
        demands = path_set.od_demands
    else:
        demands = torch.as_tensor(od_demands, dtype=torch.float32)
    if demands.numel() != len(path_set.od_pairs):
        raise ValueError("od_demands must align with the number of OD pairs in the path set.")

    path_flows = torch.zeros_like(path_set.path_costs, dtype=torch.float32)
    logits = torch.full_like(path_set.path_costs, fill_value=float("-inf"))

    for od_idx, demand in enumerate(demands):
        mask = path_set.path_od_mapping == od_idx
        if not torch.any(mask):
            continue
        od_costs = path_set.path_costs[mask]
        od_path_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        logits[od_path_indices] = -od_costs
        if demand <= 0:
            continue
        best_local = int(torch.argmin(od_costs).item())
        best_path = od_path_indices[best_local]
        path_flows[best_path] = float(demand)

    return path_flows, logits
