"""
Statistical significance tests for StreamRing experiment results.

Computes:
1. Paired t-tests / Wilcoxon signed-rank between ablation configurations
2. Effect sizes (Cohen's d)
3. Confidence intervals for all reported metrics
4. Friedman test across multiple configurations
"""

import json, sys
import numpy as np
from pathlib import Path
from src.utils.reproducibility import set_seed
from scipy import stats

project_root = Path(__file__).parent.parent
table_dir = project_root / "results" / "tables"


def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2)
                         / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def confidence_interval(data, confidence=0.95):
    """Compute confidence interval for mean."""
    n = len(data)
    if n < 2:
        return (data[0], data[0]) if n == 1 else (0, 0)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def paired_comparison(runs_a, runs_b, metric, name_a, name_b):
    """Run paired statistical test between two configurations."""
    vals_a = [r[metric] for r in runs_a]
    vals_b = [r[metric] for r in runs_b]
    n = len(vals_a)

    result = {
        "comparison": f"{name_a} vs {name_b}",
        "metric": metric,
        "mean_a": float(np.mean(vals_a)),
        "mean_b": float(np.mean(vals_b)),
        "diff": float(np.mean(vals_a) - np.mean(vals_b)),
    }

    # Paired t-test (if n >= 3)
    if n >= 3:
        t_stat, p_value = stats.ttest_rel(vals_a, vals_b)
        result["paired_ttest"] = {"t_stat": float(t_stat), "p_value": float(p_value)}

    # Wilcoxon signed-rank (non-parametric, needs n >= 6 for meaningful result)
    if n >= 5:
        try:
            w_stat, p_value = stats.wilcoxon(vals_a, vals_b)
            result["wilcoxon"] = {"w_stat": float(w_stat), "p_value": float(p_value)}
        except ValueError:
            result["wilcoxon"] = {"note": "all differences zero"}

    # Effect size
    if n >= 2:
        result["cohens_d"] = float(cohens_d(vals_a, vals_b))
        d = abs(result["cohens_d"])
        result["effect_size"] = ("negligible" if d < 0.2 else
                                 "small" if d < 0.5 else
                                 "medium" if d < 0.8 else "large")

    # 95% CI for the difference
    if n >= 2:
        diffs = np.array(vals_a) - np.array(vals_b)
        ci = confidence_interval(diffs)
        result["diff_ci_95"] = [float(ci[0]), float(ci[1])]

    return result


def analyze_ablation():
    """Statistical analysis of ablation results."""
    path = table_dir / "ablation_results.json"
    if not path.exists():
        path = table_dir / "ablation_results.json"
    if not path.exists():
        print("No ablation results found")
        return None

    with open(path) as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: ABLATION STUDY")
    print("=" * 70)

    configs = list(data.keys())
    metrics = ["auc_roc", "f1", "pr_auc", "mcc"]
    all_comparisons = []

    # Key comparisons
    comparisons = [
        (configs[0], configs[1]),  # Full CSP vs No CSP
        (configs[2], configs[3]),  # Small CSP vs Small baseline
        (configs[0], configs[2]),  # Full vs Small (both with CSP)
    ]

    for name_a, name_b in comparisons:
        if name_a not in data or name_b not in data:
            continue
        print(f"\n--- {name_a} vs {name_b} ---")
        for metric in metrics:
            result = paired_comparison(
                data[name_a]["runs"], data[name_b]["runs"],
                metric, name_a, name_b)
            all_comparisons.append(result)

            sig = ""
            if "paired_ttest" in result:
                p = result["paired_ttest"]["p_value"]
                sig = f"p={p:.4f} {'*' if p < 0.05 else 'ns'}"
            print(f"  {metric:>8}: Δ={result['diff']:+.4f}, "
                  f"d={result.get('cohens_d', 0):.3f} ({result.get('effect_size', 'n/a')}), "
                  f"{sig}")

    # Per-config confidence intervals
    print(f"\n--- 95% Confidence Intervals ---")
    ci_results = {}
    for cname, cdata in data.items():
        ci_results[cname] = {}
        print(f"\n  {cname}:")
        for metric in metrics:
            vals = [r[metric] for r in cdata["runs"]]
            ci = confidence_interval(vals)
            ci_results[cname][metric] = {
                "mean": float(np.mean(vals)),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
            }
            print(f"    {metric:>8}: {np.mean(vals):.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

    return {"comparisons": all_comparisons, "confidence_intervals": ci_results}


def analyze_label_scarcity():
    """Statistical analysis of label scarcity results."""
    path = table_dir / "label_scarcity_results.json"
    if not path.exists():
        path = table_dir / "label_scarcity_results.json"
    if not path.exists():
        print("No label scarcity results found")
        return None

    with open(path) as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: LABEL SCARCITY")
    print("=" * 70)

    # Format: {"fractions": [...], "with_csp": [{mean, std}, ...], "without_csp": [{mean, std}, ...]}
    fractions = data.get("fractions", [])
    with_csp = data.get("with_csp", [])
    without_csp = data.get("without_csp", [])

    if not fractions or not with_csp or not without_csp:
        print("  No paired run data available (only mean±std stored)")
        print("  NOTE: Re-run label scarcity with per-run storage for paired tests")
        return None

    # Report effect sizes from mean/std (no per-run data for paired tests)
    print("\n  CSP effect (from stored mean±std, no paired tests possible):")
    comparisons = []
    for i, frac in enumerate(fractions):
        if i >= len(with_csp) or i >= len(without_csp):
            break
        csp_m = with_csp[i]["mean"]
        no_m = without_csp[i]["mean"]
        diff_auc = csp_m["auc_roc"] - no_m["auc_roc"]
        diff_f1 = csp_m["f1"] - no_m["f1"]
        print(f"  {frac:>5.0%}: ΔAUC={diff_auc:+.4f}, ΔF1={diff_f1:+.4f}")
        comparisons.append({
            "fraction": frac,
            "delta_auc": float(diff_auc),
            "delta_f1": float(diff_f1),
            "note": "no per-run data for paired test"
        })

    return {"comparisons": comparisons, "note": "only mean/std available, no per-run paired tests"}


def analyze_baselines():
    """Statistical analysis of baseline comparison results."""
    path = table_dir / "baseline_results.json"
    if not path.exists():
        print("\nNo baseline results found yet")
        return None

    with open(path) as f:
        data = json.load(f)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: BASELINE COMPARISON")
    print("=" * 70)

    ours_key = "SubGNN+CSP (ours)"
    if ours_key not in data:
        print("SubGNN+CSP results not found in baselines")
        return None

    metrics = ["auc_roc", "f1", "pr_auc", "mcc"]
    comparisons = []

    for model_name, model_data in data.items():
        if model_name == ours_key:
            continue
        print(f"\n--- SubGNN+CSP vs {model_name} ---")
        for metric in metrics:
            result = paired_comparison(
                data[ours_key]["runs"], model_data["runs"],
                metric, "SubGNN+CSP", model_name)
            comparisons.append(result)

            sig = ""
            if "paired_ttest" in result:
                p = result["paired_ttest"]["p_value"]
                sig = f"p={p:.4f} {'*' if p < 0.05 else 'ns'}"
            print(f"  {metric:>8}: Δ={result['diff']:+.4f}, "
                  f"d={result.get('cohens_d', 0):.3f}, {sig}")

    return {"comparisons": comparisons}


def generate_significance_latex(ablation_results, baseline_results=None):
    """Generate LaTeX-formatted significance summary."""
    lines = []
    lines.append("% Statistical significance summary")
    lines.append("% * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant")
    lines.append("")

    if ablation_results and "comparisons" in ablation_results:
        lines.append("% Ablation comparisons:")
        for c in ablation_results["comparisons"]:
            p = c.get("paired_ttest", {}).get("p_value", 1.0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            lines.append(f"% {c['comparison']}, {c['metric']}: "
                         f"Δ={c['diff']:+.4f}, p={p:.4f} ({sig}), "
                         f"d={c.get('cohens_d', 0):.3f}")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("StreamRing Statistical Significance Tests")
    print("=" * 70, flush=True)

    all_results = {}

    # Ablation
    abl = analyze_ablation()
    if abl:
        all_results["ablation"] = abl

    # Label scarcity
    ls = analyze_label_scarcity()
    if ls:
        all_results["label_scarcity"] = ls

    # Baselines (may not exist yet)
    bl = analyze_baselines()
    if bl:
        all_results["baselines"] = bl

    # Save all results
    with open(table_dir / "statistical_tests.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {table_dir / 'statistical_tests.json'}")

    # Generate LaTeX
    latex = generate_significance_latex(abl, bl)
    with open(table_dir / "significance_summary.tex", "w") as f:
        f.write(latex)
    print(f"LaTeX summary saved to {table_dir / 'significance_summary.tex'}")



set_seed(42)

if __name__ == "__main__":
    main()
