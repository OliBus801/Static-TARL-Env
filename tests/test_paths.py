"""Tests for the path generation and AoN assignment helpers."""
from __future__ import annotations

import torch
from torch_geometric.data import Data

from simulation.paths import all_or_nothing_assignment, build_path_set


def build_toy_graph() -> Data:
    x = torch.zeros((3, 1), dtype=torch.float32)
    edge_index = torch.tensor(
        [
            [0, 0, 1],
            [1, 2, 2],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([1.0, 1.2, 1.0], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, length=lengths, num_nodes=3)


def test_path_set_contains_incidence_matrix():
    graph = build_toy_graph()
    od_matrix = torch.zeros((3, 3), dtype=torch.float32)
    od_matrix[0, 1] = 3.0
    od_matrix[0, 2] = 5.0

    path_set = build_path_set(graph, od_matrix, k=3)

    assert path_set.path_edge_incidence.shape[1] == graph.edge_index.size(1)
    assert len(path_set.paths) == path_set.path_edge_incidence.size(0)

    # The OD pair (0->1) has exactly one path: the direct edge 0->1 (edge 0).
    mask_direct = path_set.path_od_mapping == 0
    assert torch.sum(mask_direct) == 1
    assert torch.allclose(
        path_set.path_edge_incidence[mask_direct][0],
        torch.tensor([1.0, 0.0, 0.0]),
    )

    # The OD pair (0->2) should expose two distinct paths within k=3.
    mask_long = path_set.path_od_mapping == 1
    assert torch.sum(mask_long) == 2
    path_lengths = path_set.path_edge_incidence[mask_long].sum(dim=1)
    assert torch.all(path_lengths == torch.tensor([1.0, 2.0]))


def test_all_or_nothing_assignment_matches_demands():
    graph = build_toy_graph()
    od_matrix = torch.zeros((3, 3), dtype=torch.float32)
    od_matrix[0, 1] = 3.0
    od_matrix[0, 2] = 5.0

    path_set = build_path_set(graph, od_matrix, k=3)
    flows, logits = all_or_nothing_assignment(path_set)

    assert flows.numel() == path_set.num_paths
    assert logits.numel() == path_set.num_paths

    for od_idx, demand in enumerate(path_set.od_demands):
        mask = path_set.path_od_mapping == od_idx
        assert torch.isclose(flows[mask].sum(), demand)

    # For OD (0->2) the shortest path is the direct edge with cost 1.2.
    mask_long = path_set.path_od_mapping == 1
    od_costs = path_set.path_costs[mask_long]
    chosen_index = torch.argmax(logits[mask_long])
    assert torch.isclose(od_costs[chosen_index], torch.min(od_costs))
