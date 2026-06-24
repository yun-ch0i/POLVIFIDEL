"""
pilot_summary.py

Print a condition-comparison summary table from data/metrics/metrics.csv.
Runs Wilcoxon signed-rank tests (paired by image_id) for each aux condition vs baseline.

Usage:
    python pilot_summary.py
    python pilot_summary.py --metrics data/metrics/metrics.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

FOCUS_METRICS = ["polvifidel", "vifidel", "bleu4", "meteor", "rouge_l", "cider"]
CONDITIONS_ORDER = ["baseline", "textual_aux", "visual_aux", "annotated_aux"]


def wilcoxon_vs_baseline(df: pd.DataFrame, metric: str, condition: str) -> tuple[float, float]:
    """Paired Wilcoxon test: condition vs baseline on the same image_id set."""
    base = df[df["condition"] == "baseline"][["image_id", metric]].dropna()
    cond = df[df["condition"] == condition][["image_id", metric]].dropna()
    paired = base.merge(cond, on="image_id", suffixes=("_base", "_cond"))
    if len(paired) < 5:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval = stats.wilcoxon(paired[f"{metric}_cond"], paired[f"{metric}_base"])
    delta = paired[f"{metric}_cond"].mean() - paired[f"{metric}_base"].mean()
    return delta, pval


def sig_stars(pval: float) -> str:
    if np.isnan(pval):
        return ""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    if pval < 0.10:
        return "."
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", default="data/metrics/metrics.csv",
        help="Path to metrics CSV output from 05_compute_metrics.py",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    df["image_id"] = df["image_id"].astype(str)

    metrics = [m for m in FOCUS_METRICS if m in df.columns]
    conditions = [c for c in CONDITIONS_ORDER if c in df["condition"].unique()]
    models = sorted(df["model"].unique())

    for model in models:
        mdf = df[df["model"] == model]
        n_images = mdf["image_id"].nunique()
        print(f"\n{'='*70}")
        print(f"  Model: {model}   ({n_images} images, {len(mdf)} caption rows)")
        print(f"{'='*70}")

        # ── Mean ± SD table ──────────────────────────────────────────────────
        rows = []
        for cond in conditions:
            sub = mdf[mdf["condition"] == cond]
            row = {"condition": cond, "n": len(sub)}
            for m in metrics:
                vals = sub[m].dropna()
                row[m] = f"{vals.mean():.3f} ±{vals.std():.3f}" if len(vals) else "—"
            rows.append(row)

        summary = pd.DataFrame(rows).set_index("condition")
        print("\nMean ± SD by condition:")
        print(summary.to_string())

        # ── Δ vs baseline + Wilcoxon p-value ─────────────────────────────────
        print("\nΔ vs baseline  (Wilcoxon signed-rank, * p<.05, ** p<.01, *** p<.001):")
        header = f"{'condition':<20}" + "".join(f"{m:>16}" for m in metrics)
        print(header)
        print("-" * len(header))
        for cond in conditions:
            if cond == "baseline":
                continue
            row_str = f"{cond:<20}"
            for m in metrics:
                delta, pval = wilcoxon_vs_baseline(mdf, m, cond)
                if np.isnan(delta):
                    row_str += f"{'—':>16}"
                else:
                    sign = "+" if delta >= 0 else ""
                    cell = f"{sign}{delta:.3f}{sig_stars(pval)}"
                    row_str += f"{cell:>16}"
            print(row_str)

    print(f"\n{'='*70}")
    print("Significance: . p<.10  * p<.05  ** p<.01  *** p<.001")
    print("Note: small pilot (n≈20) — treat p-values as directional, not confirmatory.")


if __name__ == "__main__":
    main()
