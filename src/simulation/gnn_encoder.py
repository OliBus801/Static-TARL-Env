"""Graph neural network encoder utilities for TAP scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


@dataclass
class GNNEncoderConfig:
    """Simple configuration holder for the encoder dimensions."""

    node_in_channels: int = 2
    node_hidden_channels: int = 32
    num_gcn_layers: int = 2
    edge_in_channels: int = 3
    edge_hidden_channels: int = 32
    edge_out_channels: int = 32


class NodeEuclideanEncoder(nn.Module):
    """Project Euclidean coordinates into a latent space."""

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.mlp(coordinates)


class GNNEncoder(nn.Module):
    """Encode TAP graphs into reusable embeddings."""

    def __init__(self, config: GNNEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or GNNEncoderConfig()
        self.node_encoder = NodeEuclideanEncoder(
            self.config.node_in_channels, self.config.node_hidden_channels
        )
        self.convs = nn.ModuleList(
            [
                GCNConv(
                    self.config.node_hidden_channels,
                    self.config.node_hidden_channels,
                )
                for _ in range(self.config.num_gcn_layers)
            ]
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(
                2 * self.config.node_hidden_channels + self.config.edge_in_channels,
                self.config.edge_hidden_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                self.config.edge_hidden_channels, self.config.edge_out_channels
            ),
        )

    @property
    def node_embedding_dim(self) -> int:
        return self.config.node_hidden_channels

    @property
    def edge_embedding_dim(self) -> int:
        return self.config.edge_out_channels

    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        if data.x is None:
            raise ValueError("Graph data must contain node coordinates (x).")
        if data.edge_attr is None:
            raise ValueError("Graph data must contain edge attributes (edge_attr).")

        node_embeddings = self.node_encoder(data.x)
        for conv in self.convs:
            node_embeddings = conv(node_embeddings, data.edge_index)
            node_embeddings = F.relu(node_embeddings)

        source, target = data.edge_index
        edge_input = torch.cat(
            [node_embeddings[source], node_embeddings[target], data.edge_attr], dim=1
        )
        edge_embeddings = self.edge_mlp(edge_input)

        return {
            "node_euclidean_embeddings": node_embeddings,
            "edge_embeddings": edge_embeddings,
        }
