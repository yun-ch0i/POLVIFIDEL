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


def rq2():
    s = pd.read_csv(METRICS_DIR / "summary_covered.csv")
    val = lambda m, c: float(s[(s.model == m) & (s.condition == c)]["polvifidel"])
    gem = [val("gemini", c) for c in CONDS]
    gpt = [val("gpt4o", c) for c in CONDS]
    fig, ax = plt.subplots(figsize=(7, 5)); x = np.arange(4)
    ax.bar(x - W / 2, gem, W, label="Gemini", color=MID)
    ax.bar(x + W / 2, gpt, W, label="GPT-4o", color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(CLAB); ax.set_ylabel("POLVIFIDEL")
    ax.set_title("Fidelity by prompting condition", fontweight="bold")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGDIR / "fig_rq2_polvifidel.png"); plt.close(fig)


def rq3():
    t = pd.read_csv(METRICS_DIR / "topic_results.csv")
    def nmi(m, c):
        r = t[(t.model == m) & (t.condition == c)]
        return float(r["nmi_cell_mean"]) if len(r) else np.nan
    gem = [nmi("gemini", c) for c in CONDS]; gpt = [nmi("gpt4o", c) for c in CONDS]
    human = float(t[t.model == "nyt"]["nmi_cell_mean"])
    fig, ax = plt.subplots(figsize=(7, 5)); x = np.arange(4)
    ax.bar(x - W / 2, gem, W, label="Gemini", color=MID)
    ax.bar(x + W / 2, gpt, W, label="GPT-4o", color=NAVY)
    ax.axhline(human, color=RED, lw=2.5, ls="--")
    ax.text(3.4, human + 0.004, f"human {human:.2f}", color=RED, ha="right",
            fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(CLAB); ax.set_ylabel("Topic NMI vs. cells")
    ax.set_title("Downstream topic alignment", fontweight="bold")
    ax.legend(frameon=False, loc="upper center"); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGDIR / "fig_rq3_nmi.png"); plt.close(fig)


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
