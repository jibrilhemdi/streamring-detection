"""Smoke tests for SubGNN encoder and classifier."""

import pytest
import torch
from src.gnn_models.subgnn_encoder import SubGNNEncoder, FraudRingClassifier


@pytest.fixture
def dummy_batch():
    """Create a minimal batch of 2 subgraphs."""
    num_nodes = 10
    node_feat_dim = 32
    x = torch.randn(num_nodes, node_feat_dim)
    # Simple chain: 0-1-2-3-4 (subgraph 0), 5-6-7-8-9 (subgraph 1)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 5, 6, 7, 8],
        [1, 2, 3, 4, 6, 7, 8, 9],
    ])
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return x, edge_index, batch, node_feat_dim


def test_subgnn_encoder_output_shape(dummy_batch):
    x, edge_index, batch, feat_dim = dummy_batch
    output_dim = 64
    encoder = SubGNNEncoder(node_feat_dim=feat_dim, hidden_dim=32, output_dim=output_dim, num_layers=2)
    out = encoder(x, edge_index, batch)
    assert out.shape == (2, output_dim), f"Expected (2, {output_dim}), got {out.shape}"


def test_subgnn_encoder_with_position(dummy_batch):
    x, edge_index, batch, feat_dim = dummy_batch
    encoder = SubGNNEncoder(node_feat_dim=feat_dim, hidden_dim=32, output_dim=64, num_layers=2, num_anchors=8)
    pos_enc = torch.randn(10, 8)
    out = encoder(x, edge_index, batch, position_encoding=pos_enc)
    assert out.shape == (2, 64)


def test_fraud_ring_classifier_forward(dummy_batch):
    x, edge_index, batch, feat_dim = dummy_batch
    model = FraudRingClassifier(node_feat_dim=feat_dim, hidden_dim=32, embedding_dim=64, num_layers=2)
    logits = model(x, edge_index, batch)
    assert logits.shape == (2, 2), f"Expected (2, 2), got {logits.shape}"


def test_fraud_ring_classifier_gradients(dummy_batch):
    x, edge_index, batch, feat_dim = dummy_batch
    model = FraudRingClassifier(node_feat_dim=feat_dim, hidden_dim=32, embedding_dim=64, num_layers=2)
    logits = model(x, edge_index, batch)
    labels = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    # Check that gradients exist (position channel excluded when no position_encoding)
    for name, param in model.named_parameters():
        if param.requires_grad and "position" not in name:
            assert param.grad is not None, f"No gradient for {name}"
