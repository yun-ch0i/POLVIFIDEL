"""
01c_classify_mass_cells.py

Two-stage construction of the MASS cells (replaces the noisy keyword-based mass
files written by 01a). The elite/figure cells from 01a are left untouched.

Stage 1 (offline retrieval): from df_2025.csv, gather candidate rows that look
like a political protest — a protest-context word (WORD-BOUNDARY, so 'maga' no
longer matches 'magazine') in a US-political news_desk. Leaning-agnostic.

Stage 2 (LLM zero-shot): for each candidate, read headline+caption and classify
  {is_protest, leaning(liberal/conservative/nonpartisan), salience(high/low),
   cause, confidence}.
Rows that are real protests with a clear leaning and confidence >= threshold are
routed to D_mass_/R_mass_ × high/low. Everything else (policy articles, ambiguous,
nonpartisan) is dropped. A per-(cell,cause) cap keeps one movement (e.g. 'No
Kings') from dominating a cell.

Results are cached by image_id (data/recollected/_mass_llm_cache.csv) so re-runs
don't re-call the API.

Usage:
    python 01c_classify_mass_cells.py --dry-run          # Stage 1 only: pool size, no API
    python 01c_classify_mass_cells.py                    # full run (needs OPENAI_API_KEY)
    python 01c_classify_mass_cells.py --model gpt-4o --limit 50 --min-confidence 0.6

Writes: data/recollected/df_{D,R}_mass_{high,low}.csv  → then run 01b.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import pandas as pd
from tqdm import tqdm

from config import DATA_DIR

OUT_DIR = DATA_DIR / "recollected"
CACHE = OUT_DIR / "_mass_llm_cache.csv"

US_POLITICAL_DESKS = {"National", "Washington", "Metro", "Politics", "U.S."}

PROTEST_WORDS = [
    "protest", "protests", "protester", "protesters", "demonstration",
    "demonstrations", "demonstrator", "demonstrators", "rally", "rallies",
    "marchers", "picket", "pickets", "walkout", "sit-in", "activists",
    "rallygoers", "vigil", "vigils",
]
PROTEST_RE = re.compile(r"\b(?:" + "|".join(PROTEST_WORDS) + r")\b", re.IGNORECASE)

OUT_COLS = ["headline", "person_keywords", "multimedia_default_url", "caption",
            "news_desk", "section_name", "subsection_name", "source",
            "web_url", "keyword"]

SYSTEM_PROMPT = (
    "You classify New York Times political images from their headline and caption. "
    "You never see the image; infer only from text. Respond with strict JSON."
)

USER_TEMPLATE = """Headline: {headline}
Caption: {caption}

Decide, about the IMAGE this caption describes:
1. is_protest: true only if the image likely shows a protest, march, rally, demonstration, picket, or politically-mobilized crowd of ordinary people (NOT a portrait, official, podium, courtroom, or building).
2. leaning: the political side of the people/cause depicted — "liberal", "conservative", or "nonpartisan" (use nonpartisan if mixed, unclear, or not a US partisan cause).
3. salience: "high" for a national/large-scale movement or event; "low" for a local/community-level action (school board, town hall, local rally).
4. cause: a short label for the issue (e.g. "no kings", "abortion rights", "immigration", "book bans").
5. confidence: 0.0-1.0 for your overall judgment.

Return ONLY: {{"is_protest": bool, "leaning": str, "salience": str, "cause": str, "confidence": float}}"""


def stage1_pool(archive: str) -> pd.DataFrame:
    df = pd.read_csv(archive).dropna(subset=["multimedia_default_url", "caption"])
    df = df[df["multimedia_default_url"].astype(str).str.startswith("http")].copy()
    text = df["headline"].fillna("") + " " + df["caption"].fillna("")
    mask = text.apply(lambda t: bool(PROTEST_RE.search(t)))
    df = df[mask & df["news_desk"].isin(US_POLITICAL_DESKS)].copy()

    def cid(u):
        from pathlib import Path
        s = Path(str(u).split("?")[0]).stem
        return re.sub(r"-(jumbo|articleLarge|articleInline|master\d+|popup|"
                      r"mediumThreeByTwo\d+|videoSmall|videoLarge|sfSpan|sub)$", "", s)
    df["image_id"] = df["multimedia_default_url"].map(cid)
    return df.drop_duplicates(subset="image_id")


def classify(headline: str, caption: str, client, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                headline=str(headline)[:300], caption=str(caption)[:600])},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        seed=42,
    )
    return json.loads(resp.choices[0].message.content)


def load_cache() -> dict:
    if CACHE.exists():
        c = pd.read_csv(CACHE)
        return {str(r.image_id): r._asdict() for r in c.itertuples(index=False)}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=str(DATA_DIR / "df_2025.csv"))
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI model for classification (gpt-4o for max accuracy)")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--cap-per-cause", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="Classify only N candidates (testing)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage 1 only: report pool size, make no API calls")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = stage1_pool(args.archive)
    print(f"Stage 1 candidate pool: {len(pool)} unique protest-context images "
          f"in US-political desks")
    print(pool["news_desk"].value_counts().to_string())
    if args.dry_run:
        print("\n--dry-run: no API calls made.")
        return

    cache = load_cache()
    todo = pool[~pool["image_id"].isin(cache)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(cache)} cached, classifying {len(todo)} new...")

    if len(todo):
        from openai import OpenAI
        client = OpenAI()
        new_rows = []
        for _, r in tqdm(todo.iterrows(), total=len(todo), desc="LLM classify"):
            try:
                res = classify(r["headline"], r["caption"], client, args.model)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR {r['image_id']}: {e}")
                time.sleep(5)
                continue
            rec = {"image_id": r["image_id"],
                   "is_protest": bool(res.get("is_protest", False)),
                   "leaning": str(res.get("leaning", "nonpartisan")).lower(),
                   "salience": str(res.get("salience", "high")).lower(),
                   "cause": str(res.get("cause", "")).lower()[:40],
                   "confidence": float(res.get("confidence", 0.0))}
            new_rows.append(rec)
            cache[r["image_id"]] = rec
            if len(new_rows) % 25 == 0:
                pd.DataFrame(list(cache.values())).to_csv(CACHE, index=False)
        pd.DataFrame(list(cache.values())).to_csv(CACHE, index=False)

    # ---- Route cached classifications into the 4 mass cells ----
    cls = pd.DataFrame(list(cache.values()))
    cls = cls[cls["image_id"].isin(pool["image_id"])]
    merged = pool.merge(cls, on="image_id", how="inner")

    keep = merged[
        merged["is_protest"]
        & merged["leaning"].isin(["liberal", "conservative"])
        & merged["salience"].isin(["high", "low"])
        & (merged["confidence"] >= args.min_confidence)
    ].copy()

    lean_code = {"liberal": "D", "conservative": "R"}
    keep["cell"] = keep.apply(
        lambda r: f"{lean_code[r['leaning']]}_mass_{r['salience']}", axis=1)
    keep["keyword"] = keep["cause"]

    print(f"\nKept {len(keep)} protest images (conf >= {args.min_confidence}). By cell:")
    for cell in ["D_mass_high", "R_mass_high", "D_mass_low", "R_mass_low"]:
        sub = keep[keep["cell"] == cell]
        # diversity cap per cause
        capped = (sub.groupby("cause", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), args.cap_per_cause),
                                            random_state=args.seed)))
        for c in OUT_COLS:
            if c not in capped.columns:
                capped[c] = pd.NA
        capped[OUT_COLS].to_csv(OUT_DIR / f"df_{cell}.csv", index=False)
        top = sub["cause"].value_counts().head(4).to_dict()
        print(f"  {cell:<14} {len(sub):>4} -> {len(capped):>4} after cap   "
              f"top causes: {top}")

    print(f"\nWrote mass-cell files → {OUT_DIR}")
    print(f"Next: python 01b_consolidate_corpus.py --input-dir {OUT_DIR}")


if __name__ == "__main__":
    main()
