"""Unit tests for LSTMFraudClassifier."""
import pytest
import torch
from src.gnn_models.lstm_classifier import LSTMFraudClassifier


FEAT_DIM = 32


@pytest.fixture
def model():
    return LSTMFraudClassifier(node_feat_dim=FEAT_DIM, hidden_dim=64,
                               num_layers=1, output_dim=32, dropout=0.0)


@pytest.fixture
def dummy_batch():
    """Two subgraphs: 5 nodes each, all in one batch."""
    num_nodes = 10
    x = torch.randn(num_nodes, FEAT_DIM)
    edge_index = torch.zeros(2, 8, dtype=torch.long)  # value is irrelevant; edge_index is not read by LSTM
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return x, edge_index, batch


def test_output_shape(model, dummy_batch):
    """forward() returns [num_subgraphs, 2]."""
    x, edge_index, batch = dummy_batch
    out = model(x, edge_index, batch)
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"


def test_output_shape_single_subgraph(model):
    """Works with a single subgraph."""
    x = torch.randn(7, FEAT_DIM)
    edge_index = torch.zeros(2, 6, dtype=torch.long)
    batch = torch.zeros(7, dtype=torch.long)
    out = model(x, edge_index, batch)
    assert out.shape == (1, 2)


def test_edge_index_ignored(model, dummy_batch):
    """Output must be identical regardless of what edge_index contains."""
    x, edge_index, batch = dummy_batch
    model.eval()
    with torch.no_grad():
        out_zeros = model(x, torch.zeros_like(edge_index), batch)
        out_real = model(x, edge_index, batch)
    assert torch.allclose(out_zeros, out_real), (
        "edge_index must not be read inside forward()"
    )


def test_timestamps_affect_output(model, dummy_batch):
    """Reordering nodes by timestamp must change the output values."""
    x, edge_index, batch = dummy_batch
    model.eval()
    with torch.no_grad():
        out_natural = model(x, edge_index, batch,
                            node_timestamps=list(range(10)))
        out_reversed = model(x, edge_index, batch,
                             node_timestamps=list(range(10, 0, -1)))
    assert not torch.allclose(out_natural, out_reversed), (
        "Different timestamp orderings must produce different outputs"
    )


def test_without_timestamps(model, dummy_batch):
    """node_timestamps=None should not raise; nodes treated as unordered."""
    x, edge_index, batch = dummy_batch
    out = model(x, edge_index, batch, node_timestamps=None)
    assert out.shape == (2, 2)


def test_position_encoding_ignored(model, dummy_batch):
    """position_encoding kwarg is accepted but must not change output."""
    x, edge_index, batch = dummy_batch
    model.eval()
    pos = torch.randn(10, 16)
    with torch.no_grad():
        out_no_pos = model(x, edge_index, batch)
        out_with_pos = model(x, edge_index, batch, position_encoding=pos)
    assert torch.allclose(out_no_pos, out_with_pos), (
        "position_encoding must not be read inside forward()"
    )


def test_gradients_flow(model, dummy_batch):
    """Backward pass should produce gradients for all parameters."""
    x, edge_index, batch = dummy_batch
    out = model(x, edge_index, batch)
    labels = torch.tensor([0, 1])
    loss = torch.nn.functional.cross_entropy(out, labels)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and param.grad.abs().sum() > 0, f"Zero/no gradient for {name}"


def test_variable_subgraph_sizes(model):
    """Handles subgraphs of different sizes in the same batch."""
    # Subgraph 0: 3 nodes, Subgraph 1: 7 nodes
    x = torch.randn(10, FEAT_DIM)
    edge_index = torch.zeros(2, 4, dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    out = model(x, edge_index, batch)
    assert out.shape == (2, 2)
