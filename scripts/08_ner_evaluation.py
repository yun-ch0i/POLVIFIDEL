"""
08_ner_evaluation.py

Downstream-inference evaluation #1: Named Entity Recognition.

Tests whether prompt engineering produces captions from which a researcher could
recover the *same named entities* a human (NYT) caption mentions. For each
generated caption we run NER, compare its entity set against the NER of the
reference NYT caption (the ground truth), and score precision / recall / F1.
Higher recall under an auxiliary condition = prompting helps downstream entity
extraction.

Matching is fuzzy on normalized entity strings so "Trump" ≈ "Donald J. Trump"
and "the White House" ≈ "White House" (titles/honorifics/possessives stripped,
last-name vs full-name handled by token-subset matching).

Usage:
    python 08_ner_evaluation.py --references data/df_corpus.csv
    python 08_ner_evaluation.py --references data/df_corpus.csv --spacy-model en_core_web_trf
    python 08_ner_evaluation.py --summary-only          # re-print summary from existing CSV

Inputs:
    data/captions/{model}_{condition}.csv   - generated captions
    --references CSV                         - columns: image_id, caption (NYT originals)

Outputs:
    data/metrics/ner_eval.csv
    Columns: image_id, model, condition, n_ref, n_gen,
             precision, recall, f1,
             person_precision, person_recall, person_f1,
             ref_entities, gen_entities

Notes:
    - en_core_web_sm is the default (already a project dependency). For a stronger
      NER backbone use --spacy-model en_core_web_trf (pip install + download).
    - Ground truth = entities in the reference caption. To instead use NYT's
      authoritative `person_keywords` as PERSON ground truth, that column would
      need to be carried into the --references CSV; reference-caption NER is the
      consistent default (same reference source as 05_compute_metrics.py).
    - Checkpointing: existing (image_id, model, condition) rows are skipped.
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
import spacy
from scipy import stats
from tqdm import tqdm

from config import CAPTIONS_DIR, METRICS_DIR

NER_OUT = METRICS_DIR / "ner_eval.csv"

# Politically meaningful entity types (spaCy OntoNotes labels).
DEFAULT_TYPES = {"PERSON", "ORG", "GPE", "NORP", "FAC", "LOC", "EVENT", "LAW"}

# Tokens stripped during normalization so titles/honorifics don't block matches.
STOP_TOKENS = {
    "the", "former", "president", "vice", "sen", "senator", "rep",
    "representative", "gov", "governor", "mr", "mrs", "ms", "dr", "us",
    "secretary", "speaker", "justice", "judge", "gen", "general",
}

JACCARD_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Entity extraction + normalization
# ---------------------------------------------------------------------------

def normalize_entity(text: str) -> str:
    """Lowercase, drop possessives, punctuation, and leading titles/honorifics."""
    t = text.lower().strip()
    t = re.sub(r"['’]s\b", "", t)                 # possessive 's
    t = re.sub(r"[^\w\s]", " ", t)                     # punctuation → space
    tokens = [tok for tok in t.split() if tok not in STOP_TOKENS]
    return " ".join(tokens).strip()


def extract_entities(text: str, nlp, types: set) -> list:
    """Return deduped list of (type, normalized_text) for the selected types."""
    doc = nlp(str(text))
    seen, out = set(), []
    for ent in doc.ents:
        if ent.label_ not in types:
            continue
        norm = normalize_entity(ent.text)
        if not norm:
            continue
        key = (ent.label_, norm)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ---------------------------------------------------------------------------
# Fuzzy matching + precision/recall/F1
# ---------------------------------------------------------------------------

def entities_match(a: str, b: str) -> bool:
    """Match normalized entity strings: exact, token-subset, or Jaccard >= thresh."""
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if ta <= tb or tb <= ta:                           # last-name vs full-name
        return True
    inter, union = ta & tb, ta | tb
    return bool(union) and len(inter) / len(union) >= JACCARD_THRESHOLD


def prf(ref_norms: set, gen_norms: set) -> tuple:
    """Set-overlap precision/recall/F1 with fuzzy matching.

    Returns (precision, recall, f1); NaN where the denominator is empty.
    """
    ref, gen = list(ref_norms), list(gen_norms)
    matched_ref = sum(any(entities_match(r, g) for g in gen) for r in ref)
    matched_gen = sum(any(entities_match(g, r) for r in ref) for g in gen)

    precision = matched_gen / len(gen) if gen else float("nan")
    recall = matched_ref / len(ref) if ref else float("nan")

    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_caption(ref_ents: list, gen_ents: list) -> dict:
    """Compute overall + PERSON-only P/R/F1 from entity (type, norm) lists."""
    ref_all = {norm for _, norm in ref_ents}
    gen_all = {norm for _, norm in gen_ents}
    ref_per = {norm for lab, norm in ref_ents if lab == "PERSON"}
    gen_per = {norm for lab, norm in gen_ents if lab == "PERSON"}

    p, r, f = prf(ref_all, gen_all)
    pp, pr, pf = prf(ref_per, gen_per)
    return {
        "n_ref": len(ref_all), "n_gen": len(gen_all),
        "precision": p, "recall": r, "f1": f,
        "person_precision": pp, "person_recall": pr, "person_f1": pf,
        "ref_entities": json.dumps(sorted(ref_all)),
        "gen_entities": json.dumps(sorted(gen_all)),
    }


# ---------------------------------------------------------------------------
# I/O helpers (mirrors 05_compute_metrics.py)
# ---------------------------------------------------------------------------

def load_all_captions() -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in CAPTIONS_DIR.glob("*.csv")]
    if not dfs:
        raise FileNotFoundError(f"No caption CSVs found in {CAPTIONS_DIR}")
    return pd.concat(dfs, ignore_index=True)


def load_checkpoint() -> set:
    if NER_OUT.exists():
        df = pd.read_csv(NER_OUT)
        return set(zip(df["image_id"].astype(str), df["model"], df["condition"]))
    return set()


def append_rows(rows: list) -> None:
    df_new = pd.DataFrame(rows)
    if NER_OUT.exists():
        df_new.to_csv(NER_OUT, mode="a", header=False, index=False)
    else:
        df_new.to_csv(NER_OUT, index=False)


# ---------------------------------------------------------------------------
# Summary: Δ vs baseline by condition (paired Wilcoxon), per model
# ---------------------------------------------------------------------------

def sig(p: float) -> str:
    if np.isnan(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .10 else ""


def print_summary() -> None:
    if not NER_OUT.exists():
        print("No ner_eval.csv yet.")
        return
    df = pd.read_csv(NER_OUT)
    df["image_id"] = df["image_id"].astype(str)
    focus = ["f1", "recall", "precision", "person_f1", "person_recall"]
    order = ["baseline", "textual_aux", "visual_aux", "annotated_aux"]

    for model in sorted(df["model"].unique()):
        sub_m = df[df["model"] == model]
        print("\n" + "=" * 76)
        print(f"NER vs reference — model={model}  (n images ≈ "
              f"{sub_m['image_id'].nunique()})")
        print("=" * 76)

        # Mean by condition
        rows = []
        for cond in order:
            s = sub_m[sub_m["condition"] == cond]
            if s.empty:
                continue
            row = {"condition": cond}
            for m in focus:
                v = s[m].dropna()
                row[m] = f"{v.mean():.3f}" if len(v) else "—"
            rows.append(row)
        if rows:
            print(pd.DataFrame(rows).set_index("condition").to_string())

        # Δ vs baseline (paired Wilcoxon by image_id)
        print(f"\nΔ vs baseline (paired Wilcoxon):")
        hdr = f"{'condition':<16}" + "".join(f"{m:>16}" for m in focus)
        print(hdr)
        for cond in order:
            if cond == "baseline":
                continue
            line = f"{cond:<16}"
            for m in focus:
                base = sub_m[sub_m["condition"] == "baseline"][["image_id", m]].dropna()
                comp = sub_m[sub_m["condition"] == cond][["image_id", m]].dropna()
                paired = base.merge(comp, on="image_id", suffixes=("_b", "_c"))
                if len(paired) < 5:
                    line += f"{'n/a':>16}"
                    continue
                delta = paired[f"{m}_c"].mean() - paired[f"{m}_b"].mean()
                try:
                    _, pval = stats.wilcoxon(paired[f"{m}_c"], paired[f"{m}_b"])
                except ValueError:        # all-zero differences
                    pval = float("nan")
                cell = f"{'+' if delta >= 0 else ''}{delta:.3f}{sig(pval)}"
                line += f"{cell:>16}"
            print(line)
    print("\n. p<.10  * p<.05  ** p<.01  *** p<.001")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references",
                        help="CSV with columns image_id, caption (NYT originals)")
    parser.add_argument("--spacy-model", default="en_core_web_sm",
                        help="spaCy model for NER (en_core_web_trf for higher quality)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip computation; just re-print the summary from ner_eval.csv")
    args = parser.parse_args()

    if args.summary_only:
        print_summary()
        return
    if not args.references:
        parser.error("--references is required unless --summary-only")

    ref_df = pd.read_csv(args.references).dropna(subset=["caption"])
    ref_df["image_id"] = ref_df["image_id"].astype(str)
    references = dict(zip(ref_df["image_id"], ref_df["caption"]))

    captions_df = load_all_captions()
    captions_df["image_id"] = captions_df["image_id"].astype(str)
    captions_df = captions_df[captions_df["image_id"].isin(references)]

    done = load_checkpoint()
    remaining = captions_df[~captions_df.apply(
        lambda r: (r["image_id"], r["model"], r["condition"]) in done, axis=1
    )]
    print(f"{len(done)} rows done, {len(remaining)} remaining")

    print(f"Loading spaCy ({args.spacy_model})...")
    nlp = spacy.load(args.spacy_model)

    # Cache reference-caption entities (one NER pass per unique image).
    ref_ent_cache = {}

    batch = []
    for _, row in tqdm(remaining.iterrows(), total=len(remaining)):
        image_id = row["image_id"]
        if image_id not in ref_ent_cache:
            ref_ent_cache[image_id] = extract_entities(
                references[image_id], nlp, DEFAULT_TYPES)
        ref_ents = ref_ent_cache[image_id]
        gen_ents = extract_entities(row["caption"], nlp, DEFAULT_TYPES)

        batch.append({
            "image_id": image_id,
            "model": row["model"],
            "condition": row["condition"],
            **score_caption(ref_ents, gen_ents),
        })
        if len(batch) >= 50:
            append_rows(batch)
            batch = []
    if batch:
        append_rows(batch)

    print(f"Done. Results saved to {NER_OUT}")
    print_summary()


if __name__ == "__main__":
    main()
