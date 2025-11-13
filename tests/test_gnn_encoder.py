import pytest
import torch

from simulation.gnn_encoder import GNNEncoder, GNNEncoderConfig
from simulation.network import TapScenario


def test_encoder_output_shapes(sample_files):
    scenario = TapScenario()
    network_path, _ = sample_files
    graph = scenario.load_network_from_json(network_path)

    config = GNNEncoderConfig(
        node_in_channels=graph.x.size(1),
        node_hidden_channels=8,
        num_gcn_layers=1,
        edge_in_channels=graph.edge_attr.size(1),
        edge_hidden_channels=8,
        edge_out_channels=4,
    )
    encoder = GNNEncoder(config)

    outputs = encoder(graph)

    assert outputs["node_euclidean_embeddings"].shape == (graph.num_nodes, 8)
    assert outputs["edge_embeddings"].shape == (graph.num_edges, 4)


def test_scenario_exposes_embeddings(sample_files):
    scenario = TapScenario()
    network_path, _ = sample_files
    scenario.load_network_from_json(network_path)

    encoder = scenario.build_gnn_encoder(node_hidden_channels=4, edge_out_channels=6)
    outputs = scenario.compute_embeddings(encoder)

    assert "edge_embeddings" in outputs
    assert "node_euclidean_embeddings" in outputs
    assert scenario.embeddings["edge_embeddings"].shape == (
        scenario.graph.num_edges,
        encoder.edge_embedding_dim,
    )
    assert scenario.embeddings["node_euclidean_embeddings"].shape == (
        scenario.graph.num_nodes,
        encoder.node_embedding_dim,
    )
    assert torch.equal(outputs["edge_embeddings"], scenario.embeddings["edge_embeddings"])
