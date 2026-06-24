"""
05_compute_metrics.py

Compute reference-based metrics (BLEU, METEOR, ROUGE, CIDEr, SPICE) and the
DICTIONARY-BASED grounding metrics (VIFIDEL / HAL / POLVIFIDEL).

Grounding is lexical, not CLIP: an entity is "mentioned" iff one of its aliases
(from data/entity_dict.json, built by 03_build_entity_dictionary.py) appears in
the caption. This avoids the CLIP text-text saturation (cosine 0.28 matched
everything, so every score was 1.0/0.0) and the conflation of distinct political
entities — per the CLAUDE.md design.

  R_obj (recall)  : fraction of DETECTED entities (objects + matched faces) whose
                    aliases appear in the caption.
  HAL             : fraction of the caption's NAMED ENTITIES (spaCy NER: people,
                    orgs, places, ...) that are NOT grounded in any detected
                    entity — i.e. political-entity hallucinations. (Generic noun
                    phrases like "a room" are ignored, so HAL isn't trivially 1.)
  POLVIFIDEL      = R_obj × (1 - HAL)^β

Usage:
    python 05_compute_metrics.py --references data/df_corpus.csv
    python 05_compute_metrics.py --references data/df_corpus.csv --skip-spice
    python 05_compute_metrics.py --references data/df_corpus.csv --entity-dict data/entity_dict.json

Inputs:
    data/captions/{model}_{condition}.csv   - generated captions
    data/detections/{image_id}.json         - detection outputs
    data/entity_dict.json                    - alias dictionary (from script 03)
    --references CSV                         - columns: image_id, caption

Outputs:
    data/metrics/metrics.csv
    Columns: image_id, model, condition,
             bleu1..bleu4, meteor, rouge_l, cider, spice, vifidel, hal, polvifidel

Notes:
    - SPICE requires Java 8+; skip with --skip-spice.
    - If entity_dict.json is missing, entities fall back to deterministic aliases
      (normalized label + surname), but running script 03 first is recommended.
    - Checkpointing: existing rows in metrics.csv are skipped on re-run.
"""

import argparse
import json
import re
import string

import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm

from config import BETA, CAPTIONS_DIR, DATA_DIR, DETECTIONS_DIR, METRICS_DIR

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

METRICS_OUT = METRICS_DIR / "metrics.csv"
ENTITY_DICT_PATH = DATA_DIR / "entity_dict.json"

# Caption named-entity types whose hallucination we penalize (political fidelity).
HAL_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "NORP", "FAC", "EVENT", "LAW"}

# Stripped during normalization so titles/honorifics don't block matches.
STOP_TOKENS = {
    "the", "former", "president", "vice", "sen", "senator", "rep",
    "representative", "gov", "governor", "mr", "mrs", "ms", "dr", "us",
    "secretary", "speaker", "justice", "judge", "gen", "general",
}
JACCARD_THRESHOLD = 0.5


def normalize_entity(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"['’]s\b", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    tokens = [tok for tok in t.split() if tok not in STOP_TOKENS]
    return " ".join(tokens).strip()


def entities_match(a: str, b: str) -> bool:
    """Exact / token-subset / Jaccard match on normalized entity strings."""
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if ta <= tb or tb <= ta:
        return True
    inter, union = ta & tb, ta | tb
    return bool(union) and len(inter) / len(union) >= JACCARD_THRESHOLD


def load_entity_dict(path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  WARNING: {path} not found — using fallback aliases. "
              f"Run 03_build_entity_dictionary.py for better coverage.")
        return {}


def fallback_aliases(label: str) -> list:
    """Deterministic aliases when a label is absent from the dictionary."""
    norm = normalize_entity(label)
    toks = norm.split()
    al = {norm}
    if len(toks) > 1:
        al.add(toks[-1])      # surname / head noun
    return [a for a in al if a]


def aliases_for(label: str, entity_dict: dict) -> list:
    raw = entity_dict.get(label)
    al = [normalize_entity(a) for a in raw] if raw else fallback_aliases(label)
    return [a for a in al if a]


def alias_in_caption(alias: str, caption_lc: str) -> bool:
    """Word-boundary search of a (possibly multi-word) alias in the caption."""
    return re.search(r"\b" + re.escape(alias) + r"\b", caption_lc) is not None


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    return word_tokenize(text.lower().translate(
        str.maketrans("", "", string.punctuation)
    ))


def caption_named_entities(text: str, nlp) -> list:
    """Normalized caption named entities of the politically-relevant types."""
    out, seen = [], set()
    for ent in nlp(str(text)).ents:
        if ent.label_ not in HAL_ENTITY_TYPES:
            continue
        norm = normalize_entity(ent.text)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Reference-based metrics
# ---------------------------------------------------------------------------

def compute_bleu(hypothesis: list, reference: list) -> dict:
    smooth = SmoothingFunction().method1
    weights = {
        "bleu1": (1, 0, 0, 0),
        "bleu2": (0.5, 0.5, 0, 0),
        "bleu3": (1/3, 1/3, 1/3, 0),
        "bleu4": (0.25, 0.25, 0.25, 0.25),
    }
    return {
        k: sentence_bleu([reference], hypothesis, weights=w, smoothing_function=smooth)
        for k, w in weights.items()
    }


def compute_meteor(hypothesis: list, reference: list) -> float:
    return meteor_score([reference], hypothesis)


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def compute_cider(hypotheses: dict, references: dict) -> dict:
    from pycocoevalcap.cider.cider import Cider
    gts = {k: v if isinstance(v, list) else [v] for k, v in references.items()}
    res = {k: [v] for k, v in hypotheses.items()}
    _, scores = Cider().compute_score(gts, res)
    return dict(zip(gts.keys(), scores))


def compute_spice(hypotheses: dict, references: dict) -> dict:
    from pycocoevalcap.spice.spice import Spice
    gts = {k: v if isinstance(v, list) else [v] for k, v in references.items()}
    res = {k: [v] for k, v in hypotheses.items()}
    _, scores = Spice().compute_score(gts, res)
    return {img_id: s["All"]["f"] for img_id, s in zip(gts.keys(), scores)}


# ---------------------------------------------------------------------------
# Dictionary-based grounding: VIFIDEL / HAL / POLVIFIDEL
# ---------------------------------------------------------------------------

def load_detection(image_id: str) -> dict:
    det_path = DETECTIONS_DIR / f"{image_id}.json"
    if not det_path.exists():
        return {"objects": [], "faces": []}
    with open(det_path) as f:
        return json.load(f)


def get_detected_labels(detection: dict, fine: bool) -> list:
    """Detected entity labels for recall/grounding.

    fine=False -> GENERIC GDINO object category ("flag")  -> used by VIFIDEL.
    fine=True  -> fine-grained gallery subcategory ("thin_blue_line_flag" when
                  matched, else the generic label) -> used by POLVIFIDEL.
    Matched face identities are included in BOTH.
    """
    if fine:
        objs = [(o.get("subcategory") or o["label"]) for o in detection.get("objects", [])]
    else:
        objs = [o["label"] for o in detection.get("objects", [])]
    faces = [
        f["name"] for f in detection.get("faces", [])
        if f.get("match_status") == "matched" and f.get("name")
    ]
    return objs + faces


def compute_vifidel(caption: str, detection: dict, entity_dict: dict, fine: bool = False) -> float:
    """Object/entity recall. Default fine=False -> GENERIC-category recall (VIFIDEL).
    POLVIFIDEL calls this with fine=True for fine-grained political recall."""
    labels = get_detected_labels(detection, fine=fine)
    if not labels:
        return float("nan")
    caption_lc = caption.lower()
    covered = 0
    for label in labels:
        if any(alias_in_caption(a, caption_lc) for a in aliases_for(label, entity_dict)):
            covered += 1
    return covered / len(labels)


def compute_hal(caption: str, detection: dict, entity_dict: dict, nlp, fine: bool = True) -> float:
    """Fraction of caption named entities not grounded in any detected entity.
    Grounds against the fine-grained set (what POLVIFIDEL penalizes)."""
    mentions = caption_named_entities(caption, nlp)
    if not mentions:
        return 0.0  # no entity claims → nothing to hallucinate
    labels = get_detected_labels(detection, fine=fine)
    grounded_aliases = {a for label in labels for a in aliases_for(label, entity_dict)}
    if not grounded_aliases:
        return 1.0  # entity claims but nothing detected to support them
    ungrounded = sum(
        not any(entities_match(m, a) for a in grounded_aliases) for m in mentions
    )
    return ungrounded / len(mentions)


def compute_polvifidel(caption: str, detection: dict, entity_dict: dict, nlp,
                       beta: float = BETA) -> float:
    # Fine-grained political recall × hallucination penalty (the domain-adapted metric).
    r_obj = compute_vifidel(caption, detection, entity_dict, fine=True)
    hal = compute_hal(caption, detection, entity_dict, nlp, fine=True)
    if np.isnan(r_obj) or np.isnan(hal):
        return float("nan")
    return r_obj * (1.0 - hal) ** beta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all_captions() -> pd.DataFrame:
    dfs = [pd.read_csv(path) for path in CAPTIONS_DIR.glob("*.csv")]
    if not dfs:
        raise FileNotFoundError(f"No caption CSVs found in {CAPTIONS_DIR}")
    return pd.concat(dfs, ignore_index=True)


def load_checkpoint() -> set:
    if METRICS_OUT.exists():
        df = pd.read_csv(METRICS_OUT)
        return set(zip(df["image_id"], df["model"], df["condition"]))
    return set()


def append_rows(rows: list) -> None:
    df_new = pd.DataFrame(rows)
    if METRICS_OUT.exists():
        df_new.to_csv(METRICS_OUT, mode="a", header=False, index=False)
    else:
        df_new.to_csv(METRICS_OUT, index=False)


SUMMARY_METRICS = ["bleu4", "meteor", "rouge_l", "cider", "spice",
                   "vifidel", "hal", "polvifidel"]


def _paired_wilcoxon(cov, model, metric, condition):
    """Δ (condition − baseline) and Wilcoxon p, paired by image_id on the covered subset."""
    import warnings
    from scipy import stats
    md = cov[cov["model"] == model]
    base = md[md["condition"] == "baseline"][["image_id", metric]].dropna()
    cnd = md[md["condition"] == condition][["image_id", metric]].dropna()
    paired = base.merge(cnd, on="image_id", suffixes=("_b", "_c"))
    if len(paired) == 0:
        return float("nan"), float("nan")
    delta = paired[f"{metric}_c"].mean() - paired[f"{metric}_b"].mean()
    if len(paired) < 5:
        return delta, float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, p = stats.wilcoxon(paired[f"{metric}_c"], paired[f"{metric}_b"])
        except ValueError:                # e.g. all differences zero
            p = float("nan")
    return delta, p


def _stars(p):
    if p != p:                            # NaN
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .10 else ""


def summarize_conditions() -> None:
    """Aggregate metrics.csv by (model, condition) under TWO views:

      - ALL corpus  : aux conditions imputed = baseline for images where aux was
                      skipped (no matched-labeled entity) — the diluted, whole-corpus effect.
      - COVERED     : only images that actually received aux grounding (have a non-baseline
                      row) — the effect where the treatment truly applied.

    'Covered' is defined as image_ids appearing in any non-baseline condition, which —
    because 04 skips aux for uncovered images — is exactly the matched-labeled set.
    Writes summary_all.csv and summary_covered.csv next to metrics.csv.
    """
    if not METRICS_OUT.exists():
        print("No metrics.csv to summarize."); return
    df = pd.read_csv(METRICS_OUT)
    df["image_id"] = df["image_id"].astype(str)
    mcols = [c for c in SUMMARY_METRICS if c in df.columns]
    covered = set(df.loc[df["condition"] != "baseline", "image_id"])

    # COVERED subset: means + Δ-vs-baseline + paired Wilcoxon p (by image_id).
    cov = df[df["image_id"].isin(covered)]
    recs = []
    for (model, cond), g in cov.groupby(["model", "condition"]):
        rec = {"model": model, "condition": cond, "n": len(g)}
        for m in mcols:
            rec[m] = g[m].mean()
            if cond != "baseline":
                d, p = _paired_wilcoxon(cov, model, m, cond)
                rec[f"{m}_delta"], rec[f"{m}_p"] = d, p
        recs.append(rec)
    cov_summary = pd.DataFrame(recs).set_index(["model", "condition"]).sort_index()

    # ALL corpus, imputing aux := baseline for images where aux was skipped.
    parts = []
    for model in df["model"].unique():
        md = df[df["model"] == model]
        base = md[md["condition"] == "baseline"].set_index("image_id")[mcols]
        for cond in df["condition"].unique():
            sub = md[md["condition"] == cond].set_index("image_id")[mcols]
            if cond != "baseline":
                sub = sub.reindex(base.index).fillna(base)   # uncovered aux := baseline
            sub = sub.assign(model=model, condition=cond)
            parts.append(sub.reset_index())
    allimp = pd.concat(parts, ignore_index=True)
    all_summary = allimp.groupby(["model", "condition"])[mcols].mean()
    all_summary.insert(0, "n", allimp.groupby(["model", "condition"]).size())

    all_summary.to_csv(METRICS_DIR / "summary_all.csv")
    cov_summary.to_csv(METRICS_DIR / "summary_covered.csv")

    focus = [c for c in ("polvifidel", "vifidel", "hal") if c in mcols]
    print("\n=== Condition summary — ALL corpus (aux imputed=baseline where skipped) ===")
    print(all_summary[["n"] + focus].round(3))
    print(f"\n=== Condition summary — COVERED subset ({len(covered)} images), "
          f"Δ vs baseline (Wilcoxon: . p<.10  * <.05  ** <.01  *** <.001) ===")
    for model in sorted(cov_summary.index.get_level_values(0).unique()):
        print(f"\n  {model}:")
        for cond in cov_summary.loc[[model]].index.get_level_values(1):
            row = cov_summary.loc[(model, cond)]
            cells = []
            for m in focus:
                if cond == "baseline":
                    cells.append(f"{m}={row[m]:.3f}")
                else:
                    d, p = row.get(f"{m}_delta", float("nan")), row.get(f"{m}_p", float("nan"))
                    cells.append(f"{m}={row[m]:.3f} (Δ{d:+.3f}{_stars(p)})")
            print(f"    {cond:<14} n={int(row['n'])}  " + "  ".join(cells))
    print(f"\nSaved -> {METRICS_DIR/'summary_all.csv'}  and  {METRICS_DIR/'summary_covered.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True,
                        help="CSV with columns image_id and caption (NYT originals)")
    parser.add_argument("--entity-dict", default=str(ENTITY_DICT_PATH),
                        help="Alias dictionary from 03_build_entity_dictionary.py")
    parser.add_argument("--skip-spice", action="store_true",
                        help="Skip SPICE (requires Java)")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.references).dropna(subset=["caption"])
    ref_df["image_id"] = ref_df["image_id"].astype(str)
    references = dict(zip(ref_df["image_id"], ref_df["caption"]))

    captions_df = load_all_captions()
    captions_df["image_id"] = captions_df["image_id"].astype(str)
    captions_df = captions_df[captions_df["image_id"].isin(references)]

    done = load_checkpoint()
    remaining = captions_df[
        ~captions_df.apply(
            lambda r: (r["image_id"], r["model"], r["condition"]) in done, axis=1
        )
    ]
    print(f"{len(done)} rows done, {len(remaining)} remaining")

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")
    entity_dict = load_entity_dict(args.entity_dict)
    print(f"Entity dictionary: {len(entity_dict)} entities")

    # Pre-compute CIDEr / SPICE in bulk.
    print("Computing CIDEr in bulk...")
    hyp_dict = dict(zip(
        captions_df["image_id"] + "_" + captions_df["model"] + "_" + captions_df["condition"],
        captions_df["caption"]
    ))
    ref_dict = {
        img_id + "_" + model + "_" + cond: references[img_id]
        for _, (img_id, model, cond, *_) in captions_df[
            ["image_id", "model", "condition"]
        ].iterrows()
        if img_id in references
    }
    cider_scores = compute_cider(hyp_dict, ref_dict)

    spice_scores = {}
    if not args.skip_spice:
        print("Computing SPICE in bulk (slow, requires Java)...")
        spice_scores = compute_spice(hyp_dict, ref_dict)

    batch = []
    for _, row in tqdm(remaining.iterrows(), total=len(remaining)):
        image_id = row["image_id"]
        model = row["model"]
        condition = row["condition"]
        hypothesis = str(row["caption"])
        reference = references[image_id]

        hyp_tokens = tokenize(hypothesis)
        ref_tokens = tokenize(reference)
        key = f"{image_id}_{model}_{condition}"
        detection = load_detection(image_id)

        batch.append({
            "image_id": image_id,
            "model": model,
            "condition": condition,
            **compute_bleu(hyp_tokens, ref_tokens),
            "meteor":     compute_meteor(hyp_tokens, ref_tokens),
            "rouge_l":    compute_rouge_l(hypothesis, reference),
            "cider":      cider_scores.get(key, float("nan")),
            "spice":      spice_scores.get(key, float("nan")),
            "vifidel":    compute_vifidel(hypothesis, detection, entity_dict),
            "hal":        compute_hal(hypothesis, detection, entity_dict, nlp),
            "polvifidel": compute_polvifidel(hypothesis, detection, entity_dict, nlp),
        })

        if len(batch) >= 50:
            append_rows(batch)
            batch = []

    if batch:
        append_rows(batch)

    print(f"Done. Results saved to {METRICS_OUT}")
    summarize_conditions()


if __name__ == "__main__":
    main()
