"""Verify that all result JSON files exist and contain valid data."""

import json
import os
import pytest

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables")

EXPECTED_JSON_FILES = [
    "tier1_results.json",
    "tier2_results.json",
    "tier3_results.json",
    "ablation_results.json",
    "label_scarcity_results.json",
    "scalability_results.json",
    "cross_period_results.json",
    "streaming_results.json",
]


@pytest.mark.parametrize("filename", EXPECTED_JSON_FILES)
def test_result_file_exists_and_valid(filename):
    path = os.path.join(RESULTS_DIR, filename)
    assert os.path.exists(path), f"Missing result file: {filename}"
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, dict), f"{filename} should contain a JSON object"
    assert len(data) > 0, f"{filename} is empty"


def test_tier3_ablation_consistency():
    """Tier 3 main results should match ablation SubGNN+CSP-reg mean."""
    with open(os.path.join(RESULTS_DIR, "tier3_results.json")) as f:
        tier3 = json.load(f)
    with open(os.path.join(RESULTS_DIR, "ablation_results.json")) as f:
        ablation = json.load(f)

    csp_reg = ablation["SubGNN + CSP-reg"]["mean"]
    tier3_test = tier3["test_results"]

    for metric in ["auc_roc", "f1", "pr_auc", "mcc"]:
        assert abs(tier3_test[metric] - csp_reg[metric]) < 1e-6, (
            f"Tier 3 {metric}={tier3_test[metric]:.4f} != ablation CSP-reg {metric}={csp_reg[metric]:.4f}"
        )


def test_ablation_has_all_configs():
    with open(os.path.join(RESULTS_DIR, "ablation_results.json")) as f:
        data = json.load(f)
    expected = {"SubGNN + CSP-reg", "SubGNN (supervised)", "SubGNN-small + CSP-reg", "SubGNN-small (baseline)"}
    assert set(data.keys()) == expected


def test_label_scarcity_fractions():
    with open(os.path.join(RESULTS_DIR, "label_scarcity_results.json")) as f:
        data = json.load(f)
    assert data["fractions"] == [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    assert len(data["with_csp"]) == 6
    assert len(data["without_csp"]) == 6


def test_figures_exist():
    figures_dir = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
    for i in range(1, 17):
        for ext in ["png", "pdf"]:
            matches = [f for f in os.listdir(figures_dir) if f.startswith(f"fig{i}_") and f.endswith(f".{ext}")]
            assert len(matches) >= 1, f"Missing fig{i}_*.{ext}"


def test_rdt_results():
    path = os.path.join(RESULTS_DIR, "rdt_results.json")
    assert os.path.exists(path), "Missing rdt_results.json"
    with open(path) as f:
        data = json.load(f)
    assert "aggregate" in data
    assert 0 <= data["aggregate"]["rdt_mean"] <= 1
    for period in ["dao_hack", "attack_51_v1", "attack_51_v2"]:
        assert period in data
        assert data[period]["num_rings"] > 0


def test_statistical_tests():
    path = os.path.join(RESULTS_DIR, "statistical_tests.json")
    assert os.path.exists(path), "Missing statistical_tests.json"
    with open(path) as f:
        data = json.load(f)
    assert "ablation" in data
    assert "comparisons" in data["ablation"]
    assert len(data["ablation"]["comparisons"]) > 0
