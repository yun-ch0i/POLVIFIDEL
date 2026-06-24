"""
06b_metric_correlation.py

Metric-vs-metric correlation (no human annotations needed).

Answers the poster question "do standard automated metrics correlate with the
new POLVIFIDEL metric?" — i.e. does BLEU/ROUGE/METEOR/CIDEr/SPICE track political
fidelity, or does POLVIFIDEL capture something the standard metrics miss?

Two outputs:
  1. A full Spearman correlation MATRIX over all metrics (for a heatmap figure).
  2. A focused table of each STANDARD metric's rho vs the target metric
     (default: polvifidel), overall + by model + by condition.

The grounding components (vifidel, hal) are excluded from the focused table
because POLVIFIDEL is computed from them — correlating them would be circular —
but they remain in the full matrix for completeness.

Usage:
    python 06b_metric_correlation.py
    python 06b_metric_correlation.py --metrics data/metrics/metrics.csv --target polvifidel

Inputs:
    data/metrics/metrics.csv  (from 05_compute_metrics.py)

Outputs:
    data/metrics/metric_corr_matrix.csv     - full rho matrix (heatmap source)
    data/metrics/metric_vs_target.csv       - standard metrics vs target, long form
    Console: formatted tables
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from config import METRICS_DIR

STANDARD_METRICS = ["bleu1", "bleu2", "bleu3", "bleu4",
                    "meteor", "rouge_l", "cider", "spice"]
GROUNDING_METRICS = ["vifidel", "hal", "polvifidel"]
ALL_METRICS = STANDARD_METRICS + GROUNDING_METRICS


def spearman(x: pd.Series, y: pd.Series) -> tuple:
    """(rho, p) dropping rows where either value is NaN; NaN if n<5 or no variance."""
    mask = x.notna() & y.notna()
    if mask.sum() < 5 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan"), float("nan")
    rho, pval = stats.spearmanr(x[mask], y[mask])
    return round(float(rho), 4), round(float(pval), 4)


def sig_stars(pval: float) -> str:
    if np.isnan(pval):
        return ""
    return "***" if pval < .001 else "**" if pval < .01 else "*" if pval < .05 else ""


def corr_matrix(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Pairwise Spearman rho matrix (pairwise-complete observations)."""
    mat = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    for a in metrics:
        for b in metrics:
            mat.loc[a, b] = 1.0 if a == b else spearman(df[a], df[b])[0]
    return mat.astype(float)


def metric_vs_target(df: pd.DataFrame, standard: list, target: str,
                     group_col: str | None = None) -> pd.DataFrame:
    """rho of each standard metric vs the target metric, optionally per group."""
    rows = []
    groups = [("all", df)] if group_col is None \
        else [(g, df[df[group_col] == g]) for g in sorted(df[group_col].unique())]
    for gval, sub in groups:
        for metric in standard:
            if metric not in sub.columns:
                continue
            rho, pval = spearman(sub[metric], sub[target])
            rec = {"metric": metric, "rho": rho, "p_value": pval,
                   "sig": sig_stars(pval), "n": (sub[metric].notna() & sub[target].notna()).sum()}
            if group_col:
                rec[group_col] = gval
            rows.append(rec)
    return pd.DataFrame(rows)


def pivot_rho(df_long: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df_long = df_long.copy()
    df_long["cell"] = df_long.apply(
        lambda r: f"{r['rho']:.3f}{r['sig']}" if not np.isnan(r["rho"]) else "—", axis=1)
    return df_long.pivot(index="metric", columns=group_col, values="cell")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(METRICS_DIR / "metrics.csv"))
    parser.add_argument("--target", default="polvifidel",
                        help="Reference metric the standard metrics are correlated against")
    parser.add_argument("--output-dir", default=str(METRICS_DIR))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    present = [m for m in ALL_METRICS if m in df.columns]
    if args.target not in present:
        parser.error(f"target '{args.target}' not found in {args.metrics} "
                     f"(have: {present})")
    standard = [m for m in STANDARD_METRICS if m in present]

    print(f"\n{len(df)} rows, {df['image_id'].nunique()} images, "
          f"{df['model'].nunique()} models, {df['condition'].nunique()} conditions")
    print(f"Metrics present: {present}\nTarget: {args.target}\n")

    out_dir = METRICS_DIR

    # --- 1. Full correlation matrix (heatmap source) ---
    mat = corr_matrix(df, present)
    mat_path = out_dir / "metric_corr_matrix.csv"
    mat.round(3).to_csv(mat_path)
    print("=" * 60)
    print("Full Spearman correlation matrix")
    print("=" * 60)
    print(mat.round(2).to_string())

    # --- 2. Standard metrics vs target ---
    overall = metric_vs_target(df, standard, args.target)
    print("\n" + "=" * 60)
    print(f"Standard metrics vs {args.target} (overall)")
    print("=" * 60)
    print(overall.sort_values("rho", ascending=False).to_string(index=False))

    by_cond = metric_vs_target(df, standard, args.target, "condition")
    print("\n" + "=" * 60)
    print(f"Standard metrics vs {args.target} — by condition")
    print("=" * 60)
    print(pivot_rho(by_cond, "condition").to_string())

    by_model = metric_vs_target(df, standard, args.target, "model")
    print("\n" + "=" * 60)
    print(f"Standard metrics vs {args.target} — by model")
    print("=" * 60)
    print(pivot_rho(by_model, "model").to_string())

    # --- Save long-form results ---
    long = pd.concat([
        overall.assign(group_col="overall", group_val="all"),
        by_cond.rename(columns={"condition": "group_val"}).assign(group_col="condition"),
        by_model.rename(columns={"model": "group_val"}).assign(group_col="model"),
    ], ignore_index=True)
    vs_path = out_dir / "metric_vs_target.csv"
    long.to_csv(vs_path, index=False)

    print(f"\nMatrix saved to {mat_path}")
    print(f"Standard-vs-target results saved to {vs_path}")
    print("Note: * p<.05  ** p<.01  *** p<.001")
    print("Low rho = standard metrics miss what POLVIFIDEL captures "
          "(the motivating result).")


if __name__ == "__main__":
    main()
