"""
report_summary.py

Publication-ready reporting from data/metrics/metrics.csv:

  1. A COMBINED per-model table (POLVIFIDEL / VIFIDEL / HAL + BLEU4 / METEOR /
     ROUGE-L / CIDEr) — means with significance stars vs baseline — written as
     CSV (machine-readable, with delta+p columns), LaTeX, and Markdown.
  2. Significance-bar FIGURES: a headline POLVIFIDEL chart and a 3-panel
     fidelity chart (POLVIFIDEL / VIFIDEL / HAL), bars = mean ± SEM, with
     Wilcoxon stars above each aux bar.

All stats are on the COVERED subset (images that received aux grounding),
paired by image_id vs baseline — matching pilot_summary.py / 05's summary.

Usage:
    python report_summary.py
    python report_summary.py --metrics data/metrics/metrics.csv \
        --outdir data/metrics --figdir paper/figures
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["polvifidel", "vifidel", "hal", "bleu4", "meteor", "rouge_l", "cider"]
ORDER = ["baseline", "textual_aux", "visual_aux", "annotated_aux"]
NICE = {"polvifidel": "POLVIFIDEL", "vifidel": "VIFIDEL", "hal": "HAL",
        "bleu4": "BLEU-4", "meteor": "METEOR", "rouge_l": "ROUGE-L", "cider": "CIDEr"}


def stars(p):
    if p != p:
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .10 else ""


def paired_wilcoxon(cov, model, metric, condition):
    md = cov[cov["model"] == model]
    base = md[md["condition"] == "baseline"][["image_id", metric]].dropna()
    cnd = md[md["condition"] == condition][["image_id", metric]].dropna()
    pr = base.merge(cnd, on="image_id", suffixes=("_b", "_c"))
    if len(pr) == 0:
        return float("nan"), float("nan")
    delta = pr[f"{metric}_c"].mean() - pr[f"{metric}_b"].mean()
    if len(pr) < 5:
        return delta, float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p = stats.wilcoxon(pr[f"{metric}_c"], pr[f"{metric}_b"])
        except ValueError:
            p = float("nan")
    return delta, p


def build_report(cov, models, conds, metrics):
    rows = []
    for model in models:
        for c in conds:
            sub = cov[(cov["model"] == model) & (cov["condition"] == c)]
            rec = {"model": model, "condition": c, "n": len(sub)}
            for m in metrics:
                rec[f"{m}_mean"] = sub[m].mean()
                rec[f"{m}_sem"] = sub[m].sem()
                if c != "baseline":
                    d, p = paired_wilcoxon(cov, model, m, c)
                    rec[f"{m}_delta"], rec[f"{m}_p"] = d, p
            rows.append(rec)
    return pd.DataFrame(rows)


def display_table(rep, model, conds, metrics):
    """rows=conditions, cols=metrics, cell = 'mean' (baseline) or 'mean*' (aux)."""
    data = {}
    for c in conds:
        r = rep[(rep["model"] == model) & (rep["condition"] == c)].iloc[0]
        data[c] = {NICE[m]: (f"{r[f'{m}_mean']:.3f}" if c == "baseline"
                             else f"{r[f'{m}_mean']:.3f}{stars(r[f'{m}_p'])}")
                   for m in metrics}
    return pd.DataFrame(data).T[[NICE[m] for m in metrics]]


def to_markdown(dd, model):
    cols = list(dd.columns)
    out = [f"### {model}", "", "| condition | " + " | ".join(cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for cond, row in dd.iterrows():
        out.append(f"| {cond} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(out)


def bar_with_stars(ax, rep, metric, models, conds):
    x = np.arange(len(conds))
    w = 0.8 / len(models)
    for i, model in enumerate(models):
        means = [rep[(rep.model == model) & (rep.condition == c)][f"{metric}_mean"].iloc[0] for c in conds]
        sems = [rep[(rep.model == model) & (rep.condition == c)][f"{metric}_sem"].iloc[0] for c in conds]
        xs = x + (i - (len(models) - 1) / 2) * w
        bars = ax.bar(xs, means, w, yerr=sems, capsize=3, label=model)
        for j, c in enumerate(conds):
            if c == "baseline":
                continue
            r = rep[(rep.model == model) & (rep.condition == c)].iloc[0]
            s = stars(r[f"{metric}_p"])
            if s:
                ax.text(xs[j], means[j] + sems[j] + 0.01 * ax.get_ylim()[1], s,
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_aux", "") for c in conds], rotation=20)
    ax.set_title(NICE.get(metric, metric))
    ax.set_ylabel("mean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="data/metrics/metrics.csv")
    ap.add_argument("--outdir", default="data/metrics")
    ap.add_argument("--figdir", default="paper/figures")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df["image_id"] = df["image_id"].astype(str)
    covered = set(df.loc[df["condition"] != "baseline", "image_id"])
    cov = df[df["image_id"].isin(covered)]
    models = sorted(df["model"].unique())
    conds = [c for c in ORDER if c in df["condition"].unique()]
    metrics = [m for m in METRICS if m in df.columns]

    rep = build_report(cov, models, conds, metrics)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rep.round(4).to_csv(outdir / "combined_report.csv", index=False)

    md_parts, tex_parts = [], []
    for model in models:
        dd = display_table(rep, model, conds, metrics)
        md_parts.append(to_markdown(dd, model))
        tex_parts.append(dd.to_latex(
            caption=f"{model}: metric means by condition on the covered subset "
                    f"(n={int(rep[(rep.model==model)&(rep.condition=='textual_aux')]['n'].iloc[0])}); "
                    f"stars = paired Wilcoxon vs baseline (. p<.10 * <.05 ** <.01 *** <.001).",
            label=f"tab:{model}", escape=False))
    cov_n = len(covered)
    header = (f"# Condition comparison — covered subset ({cov_n} images), "
              f"stars = Wilcoxon vs baseline (. p<.10  * <.05  ** <.01  *** <.001)\n")
    (outdir / "combined_report.md").write_text(header + "\n\n".join(md_parts))
    (outdir / "combined_report.tex").write_text("\n\n".join(tex_parts))

    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)

    # Headline: POLVIFIDEL
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bar_with_stars(ax, rep, "polvifidel", models, conds)
    ax.legend(title="model"); fig.tight_layout()
    fig.savefig(figdir / "fig_polvifidel.png", dpi=200); plt.close(fig)

    # 3-panel fidelity
    panel = [m for m in ("polvifidel", "vifidel", "hal") if m in metrics]
    fig, axes = plt.subplots(1, len(panel), figsize=(5 * len(panel), 4.2))
    for ax, m in zip(np.atleast_1d(axes), panel):
        bar_with_stars(ax, rep, m, models, conds)
    np.atleast_1d(axes)[-1].legend(title="model"); fig.tight_layout()
    fig.savefig(figdir / "fig_fidelity_panel.png", dpi=200); plt.close(fig)

    print(header)
    for part in md_parts:
        print(part); print()
    print(f"Saved: {outdir/'combined_report.csv'}, .tex, .md")
    print(f"Figures: {figdir/'fig_polvifidel.png'}, {figdir/'fig_fidelity_panel.png'}")


if __name__ == "__main__":
    main()
