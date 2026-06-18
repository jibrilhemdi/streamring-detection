"""
Evaluation metrics for StreamRing including the novel Ring Detection Timeliness (RDT).
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve, auc,
    matthews_corrcoef, precision_score, recall_score, confusion_matrix,
)


def compute_auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area Under ROC Curve."""
    return roc_auc_score(y_true, y_scores)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 Score (binary)."""
    return f1_score(y_true, y_pred, zero_division=0)


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Area Under Precision-Recall Curve (better for imbalanced data)."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficient (robust to class imbalance)."""
    return matthews_corrcoef(y_true, y_pred)


def compute_rdt(ring_detections: list, ring_completions: list) -> float:
    """
    Ring Detection Timeliness (RDT) — NOVEL METRIC.
    
    Measures the fraction of fraud ring members detected BEFORE the ring
    completes its operation (e.g., before final cash-out).
    
    Args:
        ring_detections: List of dicts, each with:
            - ring_id: identifier for the fraud ring
            - member_id: address in the ring
            - detection_time: when StreamRing flagged this member
        ring_completions: List of dicts, each with:
            - ring_id: identifier
            - completion_time: when the ring's final transaction occurred
    
    Returns:
        RDT score in [0, 1]. Higher = earlier detection = better.
    """
    if not ring_detections or not ring_completions:
        return 0.0

    completion_times = {r["ring_id"]: r["completion_time"] for r in ring_completions}
    total_members = 0
    detected_before = 0

    for det in ring_detections:
        ring_id = det["ring_id"]
        if ring_id not in completion_times:
            continue
        total_members += 1
        if det["detection_time"] < completion_times[ring_id]:
            detected_before += 1

    return detected_before / max(total_members, 1)


def compute_rdt_weighted(ring_detections: list, ring_completions: list) -> float:
    """
    Weighted RDT — gives more credit for earlier detection.
    
    Weight = (completion_time - detection_time) / (completion_time - ring_start_time)
    Earlier detection → higher weight → higher score.
    """
    if not ring_detections or not ring_completions:
        return 0.0

    ring_info = {}
    for r in ring_completions:
        ring_info[r["ring_id"]] = {
            "completion": r["completion_time"],
            "start": r.get("start_time", r["completion_time"]),
        }

    total_weight = 0.0
    max_weight = 0.0

    for det in ring_detections:
        ring_id = det["ring_id"]
        if ring_id not in ring_info:
            continue
        info = ring_info[ring_id]
        duration = info["completion"] - info["start"]
        if duration <= 0:
            continue
        max_weight += 1.0
        if det["detection_time"] < info["completion"]:
            time_remaining = info["completion"] - det["detection_time"]
            total_weight += min(time_remaining / duration, 1.0)

    return total_weight / max(max_weight, 1)


def accuracy_at_latency(results: list, latency_budgets: list = None) -> dict:
    """
    Compute accuracy achieved within various latency budgets.
    
    Args:
        results: List of StreamRingOutput objects
        latency_budgets: List of latency thresholds in ms
    
    Returns:
        Dict mapping latency_budget → accuracy (fraction correctly classified)
    """
    if latency_budgets is None:
        latency_budgets = [5, 10, 25, 50, 100, 250, 500]

    output = {}
    for budget in latency_budgets:
        within_budget = [r for r in results if r["latency_ms"] <= budget]
        if within_budget:
            correct = sum(1 for r in within_budget
                         if r["predicted"] == r["actual"])
            output[budget] = correct / len(within_budget)
        else:
            output[budget] = 0.0
    return output


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_scores: np.ndarray) -> dict:
    """Compute all standard metrics."""
    return {
        "auc_roc": compute_auc_roc(y_true, y_scores),
        "f1": compute_f1(y_true, y_pred),
        "pr_auc": compute_pr_auc(y_true, y_scores),
        "mcc": compute_mcc(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty-print metrics table."""
    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    for key, value in metrics.items():
        print(f"  {key:>15s}: {value:.4f}")
    print(f"{'='*50}\n")
