"""
LSTMFraudClassifier — Tier 3 baseline for StreamRing paper comparison.

Receives the same 2-hop subgraph data as FraudRingClassifier (SubGNN),
but treats it as a temporally ordered node sequence instead of a graph.
edge_index is accepted for API compatibility but MUST NOT be read.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from typing import List, Optional


class LSTMFraudClassifier(nn.Module):
    """
    BiLSTM classifier over temporally sorted subgraph nodes.

    Fairness contract:
    - Receives the same x and batch as FraudRingClassifier
    - edge_index is NEVER read inside this class
    - node_timestamps (list, one float per node) controls sort order

    Note on defaults: hidden_dim=128, num_layers=1 is tuned so that param count
    (182,466 for feat_dim=32) is within 10% of SubGNN's 194,594 (ratio=6.23%).
    The parity test (test_parameter_count_parity) is the binding arbiter.
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, output_dim: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=node_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 2),
        )

    def forward(
        self,
        x: Tensor,                        # [num_nodes, feat_dim]
        edge_index: Tensor,               # accepted for API compat; NOT READ
        batch: Tensor,                    # [num_nodes] subgraph membership
        node_timestamps: Optional[List[float]] = None,  # [num_nodes] float timestamps (relabeled)
        position_encoding: Tensor = None, # accepted for API compat; NOT READ
    ) -> Tensor:
        """
        Returns logits of shape [num_subgraphs, 2].
        edge_index and position_encoding are ignored.
        """
        if batch.numel() == 0:
            return torch.zeros(0, 2, device=x.device)
        num_subgraphs = int(batch.max().item()) + 1

        sequences = []
        for sg_idx in range(num_subgraphs):
            mask = (batch == sg_idx)
            node_indices = mask.nonzero(as_tuple=True)[0]

            if node_timestamps is not None:
                ts = [node_timestamps[int(i.item())] for i in node_indices]
                order = sorted(range(len(ts)), key=lambda k: ts[k])
                node_indices = node_indices[order]

            sequences.append(x[node_indices])

        lengths = torch.tensor([s.size(0) for s in sequences],
                                dtype=torch.long, device=x.device)
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(padded, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)

        _, (h_n, _) = self.lstm(packed)  # discard per-step outputs; only final hidden states used
        h_fwd = h_n[-2]  # [B, hidden_dim]
        h_bwd = h_n[-1]  # [B, hidden_dim]
        h = torch.cat([h_fwd, h_bwd], dim=-1)  # [B, hidden_dim * 2]

        return self.head(h)  # [B, 2]
