"""
SubGNN 3-Channel Encoder for StreamRing Tier 3.
Based on Alsentzer et al. (NeurIPS 2020) with fraud-specific modifications.

Three channels capture different subgraph properties:
1. Neighborhood channel: connectivity to surrounding graph
2. Structure channel: internal topology (motifs, degree distribution)
3. Position channel: location within global graph
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool


class NeighborhoodChannel(nn.Module):
    """Captures how the subgraph connects to the rest of the graph."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINConv(mlp))
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.relu(h)
        # Pool over subgraph
        h = global_mean_pool(h, batch)
        return self.project(h)


class StructureChannel(nn.Module):
    """Captures internal topology of the subgraph."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINConv(mlp))
        self.project = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
            h = F.relu(h)
        # Use max pool to capture structural extremes
        h = global_max_pool(h, batch)
        return self.project(h)


class PositionChannel(nn.Module):
    """Captures position of subgraph within the global graph via anchor-based encoding."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_anchors: int = 16):
        super().__init__()
        self.num_anchors = num_anchors
        # Position is encoded as distances to anchor nodes
        self.encoder = nn.Sequential(
            nn.Linear(num_anchors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, position_encoding: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        position_encoding: [num_nodes, num_anchors] - distances to anchor nodes
        """
        h = self.encoder(position_encoding)
        return global_mean_pool(h, batch)


class SubGNNEncoder(nn.Module):
    """
    3-Channel SubGNN for subgraph-level fraud ring classification.
    
    Combines:
    1. Neighborhood connectivity
    2. Internal structure
    3. Global position
    
    into a unified subgraph representation.
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 128,
                 output_dim: int = 64, num_layers: int = 3,
                 num_anchors: int = 16, dropout: float = 0.2):
        super().__init__()
        channel_dim = output_dim // 3

        self.neighborhood = NeighborhoodChannel(
            node_feat_dim, hidden_dim, channel_dim, num_layers)
        self.structure = StructureChannel(
            node_feat_dim, hidden_dim, channel_dim, num_layers)
        self.position = PositionChannel(
            node_feat_dim, hidden_dim, output_dim - 2 * channel_dim, num_anchors)

        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor,
                position_encoding: torch.Tensor = None) -> torch.Tensor:
        """
        Encode a batch of subgraphs into fixed-size representations.
        
        Returns:
            embeddings: [num_subgraphs, output_dim]
        """
        h_neigh = self.neighborhood(x, edge_index, batch)
        h_struct = self.structure(x, edge_index, batch)

        if position_encoding is not None:
            h_pos = self.position(position_encoding, batch)
        else:
            h_pos = torch.zeros(
                h_neigh.size(0), self.position.encoder[-1].out_features,
                device=x.device)

        h = torch.cat([h_neigh, h_struct, h_pos], dim=-1)
        h = self.fusion(h)
        return self.layer_norm(h)


class FraudRingClassifier(nn.Module):
    """
    Full Tier 3 model: SubGNN encoder + classification head.
    Supports both supervised and contrastive training modes.
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 128,
                 embedding_dim: int = 64, num_layers: int = 3,
                 num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = SubGNNEncoder(
            node_feat_dim, hidden_dim, embedding_dim, num_layers, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def encode(self, x, edge_index, batch, position_encoding=None):
        """Get subgraph embeddings (for contrastive learning)."""
        return self.encoder(x, edge_index, batch, position_encoding)

    def forward(self, x, edge_index, batch, position_encoding=None):
        """Classify subgraphs as fraud ring or benign."""
        emb = self.encode(x, edge_index, batch, position_encoding)
        return self.classifier(emb)
