"""
01a_recollect_from_archive.py

Re-collect candidate images for each stratification cell by FILTERING the local
df_2025.csv archive (no NYT API calls). This fixes the root cause of the tiny
corpus: the original collection was capped at ~10 results per keyword, leaving
>99% of matching archive articles unused (e.g. 9,300 Trump articles → ~10 kept).

Diversity is preserved without letting one entity dominate via a per-entity cap
(--max-per-keyword): breadth comes from a broad query bank, depth is bounded per
person/issue. See the EXTERNAL VALIDITY note below for the alternative.

Matching by cell type:
  - PERSON cells  : a bank entry (a "Last, First" substring) matches the
                    person_keywords column. Each row is assigned to the first
                    bank entry it matches (so the per-keyword cap is well defined).
  - MASS cells    : an issue term appears in headline/caption AND a protest-context
                    word is present (so we get rallies/marches, not portraits).
  - ROLE cells    : a role term appears in headline/caption (looser; low-salience
                    officials are rarer).

Output: data/recollected/df_<cell>.csv  (same schema as the original per-cell
files, +keyword), which flow into 01b_consolidate_corpus.py via --input-dir.

Usage:
    python 01a_recollect_from_archive.py
    python 01a_recollect_from_archive.py --max-per-keyword 25 --seed 42
    python 01a_recollect_from_archive.py --archive data/df_2025.csv --out-dir data/recollected

EXTERNAL VALIDITY: to instead let entities appear at their natural frequency
(ecologically valid but Trump-dominated), pass --max-per-keyword 100000.

EDIT THE BANK BELOW. It is a starter seeded from the most frequent political
figures/issues actually present in df_2025.csv — extend it to taste.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from config import DATA_DIR

# ──────────────────────────────────────────────────────────────────────────────
# QUERY BANK  — edit freely. Person entries are lowercase "last, first" substrings
# matched against person_keywords; issue/role entries are matched in headline+caption.
# ──────────────────────────────────────────────────────────────────────────────

PERSON_BANK = {
    "D_elites_high": [
        "biden, joseph", "harris, kamala", "obama, barack", "newsom, gavin",
        "hochul, kathleen", "schumer, charles", "jeffries, hakeem",
        "mamdani, zohran", "adams, eric", "cuomo, andrew", "james, letitia",
        "lander, brad", "bass, karen", "walz, tim", "pritzker", "whitmer",
        "buttigieg", "sanders, bernard", "warren, elizabeth", "ocasio-cortez",
        "booker, cory", "shapiro, josh", "beshear", "padilla, alex",
        "murphy, philip", "crockett, jasmine", "khanna, ro", "fetterman",
        "klobuchar", "kelly, mark",
    ],
    "R_elites_high": [
        "trump, donald", "vance, j d", "rubio, marco", "hegseth, pete",
        "kennedy, robert f jr", "bondi, pamela", "johnson, mike", "bessent, scott",
        "patel, kashyap", "witkoff, steven", "vought, russell", "duffy, sean",
        "noem, kristi", "zeldin, lee", "lutnick, howard", "gabbard, tulsi",
        "mcmahon, linda", "blanche, todd", "bove, emil", "thune, john",
        "abbott, gregory", "desantis, ron", "cruz, ted", "hawley, josh",
        "scalise, steve", "greene, marjorie", "miller, stephen", "burgum, doug",
        "collins, susan", "graham, lindsey",
    ],
    # High-salience non-officeholders: media, business, activists, cultural, Fed.
    "N_high": [
        "musk, elon", "epstein, jeffrey", "kirk, charlie", "kimmel, jimmy",
        "colbert, stephen", "fallon, jimmy", "meyers, seth", "combs, sean",
        "zuckerberg, mark", "powell, jerome", "cook, lisa", "carlson, tucker",
        "rogan, joe", "winfrey", "swift, taylor", "bezos, jeffrey",
    ],
}

ROLE_BANK = {
    "elites_low": [
        "city council", "county commissioner", "state senator",
        "state representative", "state assembly", "state legislature",
        "attorney general", "secretary of state", "lieutenant governor",
        "comptroller", "alderman", "school board member", "county executive",
        "state lawmaker", "city councilman", "city councilwoman",
    ],
}

# NOTE: NYT captions describe SCENES, not issue labels, and protest imagery is
# scarce archive-wide (~1,674 rows). Terms below are scene-level + topical anchors
# that actually occur. Leaning is assigned by the issue's typical 2025 valence
# (e.g. ICE/deportation protests skew liberal; march-for-life skews conservative);
# generic protests are inherently ambiguous — treat mass-cell leaning as coarse.
ISSUE_BANK = {
    "D_mass_high": [
        "no kings", "hands off", "tesla takedown", "abortion", "reproductive",
        "immigrant", "immigration", "deportation", "ice raid", "climate",
        "pride", "lgbtq", "transgender", "gun control", "voting rights",
        "black lives", "racial justice", "pro-choice",
    ],
    "R_mass_high": [
        "march for life", "anti-abortion", "pro-life", "gun rights",
        "second amendment", "border security", "build the wall", "maga",
        "back the blue", "stop the steal", "election integrity",
    ],
    "D_mass_low": [
        "teachers union", "teacher strike", "school funding", "union",
        "labor", "strike", "tenant", "housing", "minimum wage", "library",
    ],
    "R_mass_low": [
        "critical race theory", "book ban", "book challenge", "parental rights",
        "parents rights", "moms for liberty", "school board", "bathroom bill",
    ],
}

# Mass cells additionally require one of these (so we get crowds, not portraits).
PROTEST_CONTEXT = [
    "protest", "rally", "march", "demonstration", "crowd", "activists",
    "demonstrators", "rallygoers", "picket", "walkout", "vigil",
]

# Output schema (matches original per-cell files + keyword).
OUT_COLS = ["headline", "person_keywords", "multimedia_default_url", "caption",
            "news_desk", "section_name", "subsection_name", "source",
            "web_url", "keyword"]


def clean_image_id(url: str) -> str:
    stem = Path(str(url).split("?")[0]).stem
    return re.sub(
        r"-(jumbo|articleLarge|articleInline|master\d+|popup|"
        r"mediumThreeByTwo\d+|videoSmall|videoLarge|sfSpan|sub)$",
        "", stem,
    )


def collect_person_cell(arc, terms, cap, seed):
    """Each row → first matching term; cap rows per term."""
    pk = arc["person_keywords"].fillna("").str.lower()
    parts = []
    for term in terms:
        hits = arc[pk.str.contains(re.escape(term), na=False)].copy()
        if hits.empty:
            continue
        hits = hits.drop_duplicates(subset="image_id")
        if len(hits) > cap:
            hits = hits.sample(n=cap, random_state=seed)
        hits["keyword"] = term
        parts.append(hits)
    return parts


def collect_text_cell(arc, terms, cap, seed, require_context):
    """Match terms in headline+caption (optionally AND a protest word); cap per term."""
    text = (arc["headline"].fillna("") + " " + arc["caption"].fillna("")).str.lower()
    ctx = text.str.contains("|".join(re.escape(c) for c in PROTEST_CONTEXT), na=False) \
        if require_context else pd.Series(True, index=arc.index)
    parts = []
    for term in terms:
        hits = arc[text.str.contains(re.escape(term), na=False) & ctx].copy()
        if hits.empty:
            continue
        hits = hits.drop_duplicates(subset="image_id")
        if len(hits) > cap:
            hits = hits.sample(n=cap, random_state=seed)
        hits["keyword"] = term
        parts.append(hits)
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=str(DATA_DIR / "df_2025.csv"))
    parser.add_argument("--out-dir", default=str(DATA_DIR / "recollected"))
    parser.add_argument("--max-per-keyword", type=int, default=20,
                        help="Cap images per person/issue (diversity control)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arc = pd.read_csv(args.archive).dropna(subset=["multimedia_default_url", "caption"])
    arc = arc[arc["multimedia_default_url"].astype(str).str.startswith("http")].copy()
    arc["image_id"] = arc["multimedia_default_url"].map(clean_image_id)
    print(f"Archive: {len(arc)} captioned rows, {arc['image_id'].nunique()} unique images\n")

    cap = args.max_per_keyword
    print(f"{'cell':<16}{'terms':>6}{'rows':>8}   per-keyword cap = {cap}")
    print("-" * 50)
    grand = 0
    for cell, terms in {**PERSON_BANK, **ROLE_BANK, **ISSUE_BANK}.items():
        if cell in PERSON_BANK:
            parts = collect_person_cell(arc, terms, cap, args.seed)
        elif cell in ROLE_BANK:
            parts = collect_text_cell(arc, terms, cap, args.seed, require_context=False)
        else:
            parts = collect_text_cell(arc, terms, cap, args.seed, require_context=True)

        if parts:
            df = pd.concat(parts, ignore_index=True)
            df = df.drop_duplicates(subset="image_id", keep="first")  # within-cell
        else:
            df = pd.DataFrame(columns=OUT_COLS)
        for c in OUT_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        df[OUT_COLS].to_csv(out_dir / f"df_{cell}.csv", index=False)
        grand += len(df)
        print(f"{cell:<16}{len(terms):>6}{len(df):>8}")

    print("-" * 50)
    print(f"{'TOTAL':<16}{'':>6}{grand:>8}  (pre cross-cell dedup)")
    print(f"\nWrote per-cell files → {out_dir}")
    print(f"Next: python 01b_consolidate_corpus.py --input-dir {out_dir}")


if __name__ == "__main__":
    main()
