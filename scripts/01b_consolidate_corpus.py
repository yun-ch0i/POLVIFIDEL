"""
01b_consolidate_corpus.py

Consolidate the per-cell query CSVs (data/df_<cell>.csv) into a single,
deduplicated, stratified corpus manifest: data/df_corpus.csv.

The per-cell files are RAW query results — the same article is often returned by
several keyword queries within a cell, so they contain heavy duplication. This
script:
  1. Reads each cell CSV and tags it with its cell label (from the filename) plus
     best-effort (leaning, actor, salience) factor columns.
  2. Drops rows without an image URL, and (by default) without a reference caption.
  3. Derives image_id via the same clean_image_id used elsewhere.
  4. Deduplicates by image_id (an image kept in the first cell it appears in,
     by CELL_ORDER), reporting cross-cell collisions.
  5. Balances to --target per cell (down-samples over-filled cells; reports the
     shortfall for under-filled cells — those need a top-up re-query in
     POLVIFIDEL_nyt_image_collection.ipynb).

It does NOT download images — that's the next step (point df_corpus.csv at the
download loop in 00_download_pilot_sample.py / the collection notebook).

Usage:
    python 01b_consolidate_corpus.py                       # target 150/cell
    python 01b_consolidate_corpus.py --target 50           # smaller corpus
    python 01b_consolidate_corpus.py --keep-uncaptioned    # don't drop caption-less rows
    python 01b_consolidate_corpus.py --output data/df_corpus.csv --seed 42

Output columns:
    image_id, caption, cell, leaning, actor, salience,
    headline, person_keywords, multimedia_default_url,
    news_desk, section_name, web_url, keyword
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from config import DATA_DIR

# Order also sets dedup priority: an image in >1 cell is kept in the first listed.
CELL_ORDER = [
    "D_elites_high", "R_elites_high", "elites_low", "N_high",
    "D_mass_high", "R_mass_high", "D_mass_low", "R_mass_low",
]

LEANING_MAP = {"D": "liberal", "R": "conservative", "N": "neutral"}

# actor_structure (not "actor") so df_corpus.csv is a drop-in --labels for 07_topic_modeling.py
OUTPUT_COLS = [
    "image_id", "caption", "cell", "leaning", "actor_structure", "salience",
    "headline", "person_keywords", "multimedia_default_url",
    "news_desk", "section_name", "web_url", "keyword",
]


def clean_image_id(url: str) -> str:
    """Same id derivation as the collection notebook / build_face_gallery.py."""
    stem = Path(str(url).split("?")[0]).stem
    return re.sub(
        r"-(jumbo|articleLarge|articleInline|master\d+|popup|"
        r"mediumThreeByTwo\d+|videoSmall|videoLarge|sfSpan|sub)$",
        "", stem,
    )


def parse_cell(cell: str) -> dict:
    """Best-effort decomposition of a cell label into factor columns."""
    tokens = cell.split("_")
    leaning = next((LEANING_MAP[t] for t in tokens if t in LEANING_MAP), "NA")
    actor = ("elite" if "elites" in tokens else
             "mass" if "mass" in tokens else "NA")
    salience = ("high" if "high" in tokens else
                "low" if "low" in tokens else "NA")
    return {"leaning": leaning, "actor_structure": actor, "salience": salience}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DATA_DIR),
                        help="Directory holding the per-cell df_<cell>.csv files")
    parser.add_argument("--target", type=int, default=150,
                        help="Target images per cell (over-filled cells down-sampled)")
    parser.add_argument("--keep-uncaptioned", action="store_true",
                        help="Keep rows with no reference caption (default: drop)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DATA_DIR / "df_corpus.csv"))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    frames = []
    for cell in CELL_ORDER:
        path = input_dir / f"df_{cell}.csv"
        if not path.exists():
            print(f"  WARNING: {path.name} not found — skipping")
            continue
        df = pd.read_csv(path)
        df["cell"] = cell
        for k, v in parse_cell(cell).items():
            df[k] = v
        frames.append(df)
    if not frames:
        parser.error("No per-cell CSVs found.")

    raw = pd.concat(frames, ignore_index=True)
    n_raw = len(raw)

    # Drop rows lacking an image URL (can't download) ...
    raw = raw.dropna(subset=["multimedia_default_url"])
    raw = raw[raw["multimedia_default_url"].astype(str).str.startswith("http")]
    n_no_url = n_raw - len(raw)

    # ... and (by default) lacking a reference caption.
    n_no_caption = int(raw["caption"].isna().sum())
    if not args.keep_uncaptioned:
        raw = raw.dropna(subset=["caption"])

    raw["image_id"] = raw["multimedia_default_url"].map(clean_image_id)

    # Order by cell priority so drop_duplicates(keep='first') respects CELL_ORDER.
    raw["cell"] = pd.Categorical(raw["cell"], categories=CELL_ORDER, ordered=True)
    raw = raw.sort_values("cell")

    collisions = int((raw.groupby("image_id")["cell"].nunique() > 1).sum())
    dedup = raw.drop_duplicates(subset="image_id", keep="first").copy()

    # Balance to target per cell.
    balanced = []
    report = []
    for cell in CELL_ORDER:
        sub = dedup[dedup["cell"] == cell]
        n = len(sub)
        if n > args.target:
            sub = sub.sample(n=args.target, random_state=args.seed)
            status = f"down-sampled {n}→{args.target}"
        elif n < args.target:
            status = f"SHORT by {args.target - n}  (re-query needed)"
        else:
            status = "on target"
        balanced.append(sub)
        report.append((cell, n, min(n, args.target), status))

    corpus = pd.concat(balanced, ignore_index=True)
    corpus["cell"] = corpus["cell"].astype(str)
    for c in OUTPUT_COLS:
        if c not in corpus.columns:
            corpus[c] = pd.NA
    corpus = corpus[OUTPUT_COLS]
    corpus.to_csv(args.output, index=False)

    # ---- Report ----
    print(f"\nRaw rows across {len(frames)} cells : {n_raw}")
    print(f"Dropped (no image URL)            : {n_no_url}")
    print(f"Rows with no caption              : {n_no_caption}"
          f"{'  (dropped)' if not args.keep_uncaptioned else '  (kept)'}")
    print(f"Cross-cell duplicate images       : {collisions}")
    print(f"Unique usable images              : {len(dedup)}")
    print(f"\n{'cell':<16}{'unique':>8}{'kept':>7}   status")
    print("-" * 60)
    total_have, total_short = 0, 0
    for cell, have, kept, status in report:
        total_have += have
        total_short += max(0, args.target - have)
        print(f"{cell:<16}{have:>8}{kept:>7}   {status}")
    print("-" * 60)
    print(f"{'TOTAL':<16}{total_have:>8}{len(corpus):>7}")
    print(f"\nTarget {args.target}/cell × {len(CELL_ORDER)} = {args.target*len(CELL_ORDER)}")
    print(f"Still need ~{total_short} more unique images to fill all cells.")
    print(f"\nCorpus manifest written → {args.output}  ({len(corpus)} rows)")


if __name__ == "__main__":
    main()
