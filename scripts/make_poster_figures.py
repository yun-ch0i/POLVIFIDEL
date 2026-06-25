"""
make_poster_figures.py

Generate the poster result charts (paper/figures/fig_rq2/3/4_*.png) from the
metric CSVs, styled to the poster palette. Run from the repo root:

    python scripts/make_poster_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import METRICS_DIR

FIGDIR = METRICS_DIR.parent.parent / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

NAVY, MID, RED, GREY = "#1F3A5F", "#2E6BA6", "#B23A2E", "#9AA7B4"
plt.rcParams.update({
    "font.size": 17, "axes.edgecolor": "#444", "axes.linewidth": 1.2,
    "font.family": "DejaVu Sans", "savefig.dpi": 200,
    "figure.constrained_layout.use": True,
})

CONDS = ["baseline", "textual_aux", "visual_aux", "annotated_aux"]
CLAB = ["Base", "Text", "Visual", "Annot"]
W = 0.38

# Four models: API (blues) vs. open-source (warm) so the split reads visually.
MODELS = ["gemini", "gpt4o", "internvl", "qwen"]
MLAB = ["Gemini", "GPT-4o", "InternVL", "Qwen"]
MCOLORS = ["#5B9BD5", "#1F3A5F", "#C0703A", "#6F8F3F"]


def _grouped(values_by_model, ylabel, title, fname, human=None):
    """values_by_model: dict model -> [4 condition values]. Grouped bar chart."""
    fig, ax = plt.subplots(figsize=(8.2, 5)); x = np.arange(len(CONDS))
    n = len(MODELS); bw = 0.8 / n
    for i, m in enumerate(MODELS):
        off = (i - (n - 1) / 2) * bw
        ax.bar(x + off, values_by_model[m], bw, label=MLAB[i], color=MCOLORS[i])
    if human is not None:
        ax.axhline(human, color=RED, lw=2.5, ls="--")
        ax.text(len(CONDS) - 0.55, human + 0.004, f"human {human:.2f}",
                color=RED, ha="right", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CLAB); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGDIR / fname); plt.close(fig)


def rq2():
    d = pd.read_csv(METRICS_DIR / "metrics.csv")
    piv = d.pivot_table(index="model", columns="condition", values="polvifidel", aggfunc="mean")
    vals = {m: [float(piv.loc[m, c]) for c in CONDS] for m in MODELS}
    _grouped(vals, "POLVIFIDEL", "Fidelity by prompting condition", "fig_rq2_polvifidel.png")


def rq3():
    t = pd.read_csv(METRICS_DIR / "topic_results.csv")
    def nmi(m, c):
        r = t[(t.model == m) & (t.condition == c)]
        return float(r["nmi_cell_mean"]) if len(r) else np.nan
    vals = {m: [nmi(m, c) for c in CONDS] for m in MODELS}
    human = float(t[t.model == "nyt"]["nmi_cell_mean"])
    _grouped(vals, "Topic NMI vs. cells", "Downstream topic alignment",
             "fig_rq3_nmi.png", human=human)


def rq4():
    p = pd.read_csv(METRICS_DIR / "native_vs_caption_probe.csv")
    val = lambda rep, tg: float(p[(p.representation == rep) & (p.target == tg)]["f1_macro"])
    is_vlm = p.representation.str.startswith("caption:") & (p.representation != "caption:nyt_reference")
    vlm_cell = p[is_vlm & (p.target == "cell")]["f1_macro"].mean()
    vlm_lean = p[is_vlm & (p.target == "leaning")]["f1_macro"].mean()
    reps = ["Human\ncaption", "Image-\nnative", "VLM\ncaption"]
    cell = [val("caption:nyt_reference", "cell"), val("image_clip", "cell"), vlm_cell]
    lean = [val("caption:nyt_reference", "leaning"), val("image_clip", "leaning"), vlm_lean]
    fig, ax = plt.subplots(figsize=(7, 5)); x = np.arange(3)
    ax.bar(x - W / 2, cell, W, label="cell (8-way)", color=NAVY)
    ax.bar(x + W / 2, lean, W, label="leaning (3-way)", color=RED)
    ax.axhline(0.038, color=GREY, lw=1.5, ls=":")   # cell chance
    ax.axhline(0.222, color=GREY, lw=1.5, ls=":")   # leaning chance
    ax.set_xticks(x); ax.set_xticklabels(reps)
    ax.set_ylabel("Recoverable signal (macro-F1)"); ax.set_ylim(0, 0.8)
    ax.set_title("Image-native > VLM caption", fontweight="bold")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGDIR / "fig_rq4_probe.png"); plt.close(fig)


if __name__ == "__main__":
    rq2(); rq3(); rq4()
    print(f"Wrote poster figures to {FIGDIR}")
