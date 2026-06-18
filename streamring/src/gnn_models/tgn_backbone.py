"""
TGN (Temporal Graph Network) backbone for StreamRing Tier 2.
Based on Rossi et al. (2020) with modifications for incremental inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class TimeEncoder(nn.Module):
    """Encode time differences using learnable Fourier features."""

    def __init__(self, time_dim: int):
        super().__init__()
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float64)))
            .float().reshape(time_dim, 1)
        )
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        return torch.cos(self.w(t))


class MemoryModule(nn.Module):
    """
    Per-node memory (GRU-based) tracking historical interactions.
    Each node maintains a compressed state vector updated after each event.
    """

    def __init__(self, num_nodes: int, memory_dim: int, message_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Node memory state
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))

        # GRU updater
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.memory[node_ids]

    def update_memory(self, node_ids: torch.Tensor, messages: torch.Tensor,
                      timestamps: torch.Tensor):
        """Update memory for specified nodes with new messages."""
        unique_ids, inverse = torch.unique(node_ids, return_inverse=True)
        # Aggregate messages for same node (last message wins)
        unique_messages = torch.zeros(
            unique_ids.size(0), messages.size(1), device=messages.device)
        unique_messages.scatter_(0, inverse.unsqueeze(1).expand_as(messages), messages)

        updated = self.gru(unique_messages, self.memory[unique_ids])
        self.memory[unique_ids] = updated.detach()
        self.last_update[unique_ids] = timestamps[
            torch.unique(inverse, return_inverse=False)
        ] if len(timestamps) == len(unique_ids) else self.last_update[unique_ids]

    def reset(self):
        self.memory.zero_()
        self.last_update.zero_()


class TemporalAttentionLayer(nn.Module):
    """Temporal attention over neighbors with time-aware scoring."""

    def __init__(self, input_dim: int, time_dim: int, output_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.time_encoder = TimeEncoder(time_dim)

        self.query = nn.Linear(input_dim + time_dim, output_dim)
        self.key = nn.Linear(input_dim + time_dim, output_dim)
        self.value = nn.Linear(input_dim + time_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, query_emb: torch.Tensor, key_emb: torch.Tensor,
                time_diffs: torch.Tensor, edge_features: torch.Tensor = None):
        """
        Args:
            query_emb: [batch, dim] - target node embeddings
            key_emb: [batch, num_neighbors, dim] - neighbor embeddings
            time_diffs: [batch, num_neighbors] - time since interaction
        """
        time_enc = self.time_encoder(time_diffs)

        # Concatenate with time encoding
        q = self.query(torch.cat([query_emb.unsqueeze(1).expand_as(key_emb),
                                   time_enc], dim=-1))
        k = self.key(torch.cat([key_emb, time_enc], dim=-1))
        v = self.value(torch.cat([key_emb, time_enc], dim=-1))

        # Multi-head attention
        d_k = q.size(-1) // self.num_heads
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.mean(dim=1)  # Aggregate over neighbors
        out = self.out_proj(out)
        return self.layer_norm(out + query_emb)


class TGNBackbone(nn.Module):
    """
    Full TGN model for StreamRing Tier 2.
    
    Components:
    1. Memory module (per-node GRU state)
    2. Temporal attention for neighbor aggregation
    3. Embedding computation via stacked temporal attention layers
    """

    def __init__(self, num_nodes: int, node_feat_dim: int, edge_feat_dim: int,
                 memory_dim: int = 128, embedding_dim: int = 128,
                 time_dim: int = 64, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim

        # Memory
        message_dim = memory_dim * 2 + edge_feat_dim + time_dim
        self.memory = MemoryModule(num_nodes, memory_dim, message_dim)
        self.time_encoder = TimeEncoder(time_dim)

        # Message function
        self.message_fn = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_feat_dim + time_dim, message_dim),
            nn.ReLU(),
        )

        # Embedding layers
        self.attention_layers = nn.ModuleList([
            TemporalAttentionLayer(
                input_dim=memory_dim if i == 0 else embedding_dim,
                time_dim=time_dim,
                output_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        # Node feature projection
        self.node_proj = nn.Linear(node_feat_dim, memory_dim)

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

    def compute_embedding(self, node_ids: torch.Tensor,
                          neighbor_ids: torch.Tensor,
                          time_diffs: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings via temporal attention over neighbors."""
        node_memory = self.memory.get_memory(node_ids)
        neighbor_memory = self.memory.get_memory(neighbor_ids.view(-1)).view(
            *neighbor_ids.shape, -1)

        h = node_memory
        for layer in self.attention_layers:
            h = layer(h, neighbor_memory, time_diffs)
        return h

    def forward(self, src_ids: torch.Tensor, dst_ids: torch.Tensor,
                src_neighbors: torch.Tensor, dst_neighbors: torch.Tensor,
                src_time_diffs: torch.Tensor, dst_time_diffs: torch.Tensor,
                edge_features: torch.Tensor, timestamps: torch.Tensor):
        """
        Process a batch of edges and return fraud scores.
        
        Returns:
            scores: [batch_size] tensor of fraud probabilities
        """
        # Compute embeddings
        src_emb = self.compute_embedding(src_ids, src_neighbors, src_time_diffs)
        dst_emb = self.compute_embedding(dst_ids, dst_neighbors, dst_time_diffs)

        # Compute messages for memory update
        src_memory = self.memory.get_memory(src_ids)
        dst_memory = self.memory.get_memory(dst_ids)
        time_enc = self.time_encoder(timestamps)

        src_msg = self.message_fn(
            torch.cat([src_memory, dst_memory, edge_features, time_enc], dim=-1))
        dst_msg = self.message_fn(
            torch.cat([dst_memory, src_memory, edge_features, time_enc], dim=-1))

        # Update memory
        self.memory.update_memory(src_ids, src_msg, timestamps)
        self.memory.update_memory(dst_ids, dst_msg, timestamps)

        # Classification on combined embedding
        combined = src_emb + dst_emb  # Simple combination; can also use concat
        scores = torch.sigmoid(self.classifier(combined)).squeeze(-1)
        return scores


import numpy as np  # Required for TimeEncoder
