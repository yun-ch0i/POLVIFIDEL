# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

POLVIFIDEL is a dissertation research project evaluating and improving the **political fidelity** of vision-language model (VLM)-generated image captions. It introduces a domain-adapted reference-free metric (POLVIFIDEL, extending VIFIDEL) and tests auxiliary prompting strategies to improve VLM caption quality on political news images from the New York Times.

## Codebase Structure

```
scripts/
├── config.py                         # central paths + experiment settings (env-aware)
├── 01_collect_NYT_images.py          # NYT API data collection (prototype)
├── POLVIFIDEL_nyt_image_collection.ipynb  # working collection notebook (Colab)
├── 02_detect_entities.py             # Grounding-DINO + face recognition → data/detections/
├── 03_build_entity_dictionary.py     # LLM-based paraphrase generation for alignment dicts
├── 04_generate_captions_api.py       # GPT + Gemini via Sandbox/Portkey (run locally, API calls only)
├── 04_generate_captions_cluster.py   # InternVL / Qwen (run on cluster, GPU)
├── 05_compute_metrics.py             # POLVIFIDEL, VIFIDEL, BLEU, ROUGE, etc.
└── cluster/
    ├── run_detection.slurm           # SLURM: batch entity detection
    └── run_vlm_inference.slurm       # SLURM: array job, one task per open-source model
data/
├── images/        # downloaded NYT images (gitignored)
├── detections/    # per-image detection JSON outputs (gitignored)
├── captions/      # per-model caption CSVs (gitignored)
├── metrics/       # computed metric scores (gitignored)
└── df_2025.csv    # full 2025 NYT archive metadata (gitignored)
paper/
├── polvifidel_paper.tex   # primary manuscript (xelatex)
├── polvifidel_paper.Rmd   # original source, kept for reference
└── polvifidel.bib
```

`scripts/config.py` is the single source of truth for paths and model IDs. It reads `POLVIFIDEL_ROOT` from the environment (set in SLURM scripts) and falls back to the repo root locally.

## Languages and Environments

**Python (data collection):** The notebook is designed for **Google Colab**. API keys are loaded via `google.colab.userdata`. The NYT API key should be stored as a Colab secret named appropriately and accessed via `userdata.get(...)`.

**R (analysis and paper):** RStudio project (`POLVIFIDEL.Rproj`), 2-space indentation. The paper is authored in `paper/polvifidel_paper.tex` (plain LaTeX). Compile with:
```bash
cd paper && xelatex polvifidel_paper.tex && bibtex polvifidel_paper && xelatex polvifidel_paper.tex && xelatex polvifidel_paper.tex
```
Requires `xelatex` and Times New Roman font. The `.Rmd` source is kept for reference but is no longer the primary source.

## Data Collection Logic

The notebook stratifies 1,200 NYT images across 8 cells: `(leaning, actor_structure, salience)` — e.g., `("liberal", "elite", "high")`. Target: 150 images per cell. Two retrieval strategies:
- **Elite/high-salience:** identity-based queries (person names)
- **Elite/low-salience & mass:** role/keyword-based queries from `QUERY_BANK`

The main data file `data/df_2025.csv` contains ~47k articles from the full 2025 NYT archive with columns: `headline`, `person_keywords`, `multimedia_default_url`, `caption`, `news_desk`, `section_name`, `subsection_name`, `source`, `web_url`.

## Key Design Decisions

- POLVIFIDEL metric = `R_obj × (1 - HAL)^β` where `R_obj` is object recall and `HAL` is a weighted hallucination penalty; `β ≥ 1` controls hallucination severity
- Entity alignment uses **dictionary-based matching** (not CLIP) to avoid conflating politically distinct entities
- Four VLMs benchmarked — proprietary via AI Sandbox/Portkey (paid, $250/mo cap): GPT-4o-mini, Gemini 3.1 Pro; open-source self-hosted on cluster GPU (free): InternVL2-8B, Qwen2.5-VL-7B. Model IDs live in `scripts/config.py`. (LLaVA-1.6 dropped — single-image model unreliable for multi-image visual_aux.)
- Three auxiliary prompting conditions: Visual Aux (cropped images), Textual Aux (structured text), Annotated Aux (annotated image)
