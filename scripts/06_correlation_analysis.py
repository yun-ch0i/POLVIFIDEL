"""
06_correlation_analysis.py

Compute Spearman correlations between each automatic metric and average
human fidelity judgments, following the evaluation design in the paper.

Usage:
    python 06_correlation_analysis.py \
        --metrics   data/metrics/metrics.csv \
        --human     data/human_annotations.csv \
        --output    data/metrics/correlations.csv

Expected columns:
    metrics CSV   : image_id, model, condition,
                    bleu1, bleu2, bleu3, bleu4, meteor, rouge_l,
                    cider, spice, vifidel, polvifidel  (polvifidel optional)
    human CSV     : image_id, model, condition,
                    rater_1, rater_2, rater_3
                    -- OR --
                    image_id, model, condition, human_score  (pre-averaged)

Outputs:
    data/metrics/correlations.csv   - full results table
    Console: formatted summary tables
"""

import argparse
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Metric columns (add polvifidel once computed)
# ---------------------------------------------------------------------------

METRIC_COLS = ["bleu1", "bleu2", "bleu3", "bleu4",
               "meteor", "rouge_l", "cider", "spice",
               "vifidel", "polvifidel"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spearman(x: pd.Series, y: pd.Series) -> tuple:
    """Return (rho, p-value) dropping rows where either value is NaN."""
    mask = x.notna() & y.notna()
    if mask.sum() < 5:
        return float("nan"), float("nan")
    rho, pval = stats.spearmanr(x[mask], y[mask])
    return round(float(rho), 4), round(float(pval), 4)


def sig_stars(pval: float) -> str:
    if np.isnan(pval):
        return ""
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def load_human(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rater_cols = [c for c in df.columns if c.startswith("rater_")]
    if rater_cols:
        df["human_score"] = df[rater_cols].mean(axis=1)
    if "human_score" not in df.columns:
        raise ValueError(
            "Human annotation CSV must have either 'human_score' or "
            "'rater_1', 'rater_2', ... columns."
        )
    return df[["image_id", "model", "condition", "human_score"]]


# ---------------------------------------------------------------------------
# Correlation tables
# ---------------------------------------------------------------------------

def overall_correlations(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Single row per metric: rho and p-value pooling all observations."""
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        rho, pval = spearman(df[metric], df["human_score"])
        rows.append({"metric": metric, "rho": rho, "p_value": pval,
                     "sig": sig_stars(pval), "n": df[metric].notna().sum()})
    return pd.DataFrame(rows).sort_values("rho", ascending=False)


def correlations_by_group(
    df: pd.DataFrame, metrics: list, group_col: str
) -> pd.DataFrame:
    """Rho for each metric × each group level."""
    rows = []
    for group_val in sorted(df[group_col].unique()):
        sub = df[df[group_col] == group_val]
        for metric in metrics:
            if metric not in sub.columns:
                continue
            rho, pval = spearman(sub[metric], sub["human_score"])
            rows.append({group_col: group_val, "metric": metric,
                         "rho": rho, "p_value": pval, "sig": sig_stars(pval),
                         "n": sub[metric].notna().sum()})
    return pd.DataFrame(rows)


def pivot_correlations(df_long: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Wide-format table: rows = metrics, columns = group levels."""
    df_long["rho_str"] = df_long.apply(
        lambda r: f"{r['rho']:.3f}{r['sig']}" if not np.isnan(r["rho"]) else "—",
        axis=1,
    )
    return df_long.pivot(index="metric", columns=group_col, values="rho_str")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--human",   required=True)
    parser.add_argument("--output",  default="data/metrics/correlations.csv")
    args = parser.parse_args()

    metrics_df = pd.read_csv(args.metrics)
    human_df   = load_human(args.human)

    df = metrics_df.merge(human_df, on=["image_id", "model", "condition"], how="inner")
    metrics = [m for m in METRIC_COLS if m in df.columns]

    print(f"\nMerged dataset: {len(df)} rows, {df['image_id'].nunique()} images, "
          f"{df['model'].nunique()} models, {df['condition'].nunique()} conditions\n")

    # --- Overall ---
    overall = overall_correlations(df, metrics)
    print("=" * 55)
    print("Overall Spearman correlations with human fidelity")
    print("=" * 55)
    print(overall.to_string(index=False))

    # --- By model ---
    by_model_long = correlations_by_group(df, metrics, "model")
    by_model = pivot_correlations(by_model_long, "model")
    print("\n" + "=" * 55)
    print("Correlations by model")
    print("=" * 55)
    print(by_model.to_string())

    # --- By condition ---
    by_cond_long = correlations_by_group(df, metrics, "condition")
    by_cond = pivot_correlations(by_cond_long, "condition")
    print("\n" + "=" * 55)
    print("Correlations by prompting condition")
    print("=" * 55)
    print(by_cond.to_string())

    # --- Save full results ---
    out_path = args.output
    full = pd.concat(
        [
            overall.assign(group_col="overall", group_val="all"),
            by_model_long.rename(columns={"model": "group_val"})
                         .assign(group_col="model"),
            by_cond_long.rename(columns={"condition": "group_val"})
                        .assign(group_col="condition"),
        ],
        ignore_index=True,
    )
    full.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")
    print("Note: * p<.05  ** p<.01  *** p<.001")


if __name__ == "__main__":
    main()
