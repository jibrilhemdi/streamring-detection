"""Smoke tests for evaluation metrics."""

import numpy as np
from src.evaluation.metrics import compute_auc_roc, compute_f1, compute_pr_auc, compute_mcc, compute_rdt


def test_auc_roc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_auc_roc(y_true, y_scores) == 1.0


def test_auc_roc_random():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=1000)
    y_scores = np.random.rand(1000)
    auc = compute_auc_roc(y_true, y_scores)
    assert 0.4 < auc < 0.6, f"Random scores should give AUC ~0.5, got {auc}"


def test_f1_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert compute_f1(y_true, y_pred) == 1.0


def test_pr_auc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_pr_auc(y_true, y_scores) == 1.0


def test_mcc_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert compute_mcc(y_true, y_pred) == 1.0


def test_rdt_empty():
    assert compute_rdt([], []) == 0.0


def test_rdt_all_early():
    detections = [
        {"ring_id": "r1", "member_id": "a1", "detection_time": 100},
        {"ring_id": "r1", "member_id": "a2", "detection_time": 110},
    ]
    completions = [{"ring_id": "r1", "completion_time": 200}]
    rdt = compute_rdt(detections, completions)
    assert rdt == 1.0, f"All detected before completion, expected RDT=1.0, got {rdt}"
