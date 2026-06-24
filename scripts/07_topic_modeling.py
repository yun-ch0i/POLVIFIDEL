"""
07_topic_modeling.py

Run BERTopic on captions from each model × condition, then evaluate quality
using two complementary measures:

  1. CV coherence   — intrinsic; measures within-topic word co-occurrence.
                      Higher = more semantically coherent topics.
  2. NMI            — extrinsic; measures how well recovered topics align with
                      the known experimental stratification (leaning ×
                      actor_structure × salience). Higher = topic structure
                      better captures the political categories in the corpus.

Also runs on the original NYT captions as a human-written baseline.

Usage:
    python 07_topic_modeling.py \
        --captions  data/captions/ \
        --labels    data/sample_labels.csv \
        --refs      data/df_sample.csv \
        --output    data/metrics/topic_results.csv

Expected columns:
    labels CSV  : image_id, leaning, actor_structure, salience
    refs CSV    : image_id, caption  (NYT originals)

Each run is refit over several random seeds (--n-seeds, default 10) and the
metrics reported as mean ± SD, since a single BERTopic fit is sensitive to the
UMAP seed. Captions are also cleaned of boilerplate/refusal phrases (see
BOILERPLATE_STOPWORDS) so topics reflect political content, not caption style.

Outputs:
    data/metrics/topic_results.csv          — summary: mean ± SD per model×condition
    data/metrics/topic_results_byseed.csv   — long form: one row per run×seed
    data/topics/{label}.csv                 — reference-seed per-image assignments
    data/topics/{label}_words.csv           — reference-seed top words per topic
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

from config import CAPTIONS_DIR, METRICS_DIR

warnings.filterwarnings("ignore", category=FutureWarning)

TOPICS_DIR = METRICS_DIR.parent / "topics"
TOPICS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
DEFAULT_N_SEEDS = 10   # robustness: refit each run over this many seeds, report mean ± SD

# Boilerplate / refusal phrases that the VLM captions emit regardless of image
# content. Left in, they form topics of their own ("image shows ...", "unable to
# identify ...") and distort the topic structure we want to measure. We strip
# only true meta-description and hedging/refusal filler — NOT content words like
# people/crowd/man/woman/protest/police, which carry the actor/salience signal.
# CountVectorizer removes these tokens before building n-grams, so bigrams such
# as "image shows" / "unable identify" disappear along with the unigrams.
BOILERPLATE_STOPWORDS = {
    # image-medium meta description
    "image", "images", "photo", "photos", "photograph", "photographs",
    "picture", "pictures", "shows", "show", "shown", "showing", "depicts",
    "depict", "depicted", "depicting", "depiction", "features", "feature",
    "featured", "featuring", "captures", "capture", "captured", "capturing",
    "appears", "appear", "appearing", "portrays", "portray", "portrayed",
    "scene", "setting", "background", "foreground", "visible", "seen",
    # Gemini grounding artifacts ("based image", "provided image")
    "based", "provided", "discernible", "factual", "knowledge",
    # refusal / hedging
    "unable", "identify", "identified", "identifying", "identification",
    "cannot", "unidentified", "unclear", "specific", "likely",
    "suggesting", "suggests", "suggest",
}
STOPWORDS = list(ENGLISH_STOP_WORDS | BOILERPLATE_STOPWORDS)

# BERTopic settings — fixed across all runs for fair comparison.
# min_cluster_size=15 is reasonable for a 1,200-document corpus.
HDBSCAN_KWARGS = dict(min_cluster_size=15, min_samples=5,
                      metric="euclidean", prediction_data=True)
# NOTE: BERTopic reuses this vectorizer for its c-TF-IDF step, where each
# *topic* (not each caption) is one document. min_df must therefore stay <=
# the smallest topic count any run produces, or sklearn raises
# "max_df corresponds to < documents than min_df" when a run collapses to
# fewer than min_df topics. min_df=1 is safe for that step; rare-term
# filtering still happens via stop_words + the c-TF-IDF weighting.
VECTORIZER_KWARGS = dict(stop_words=STOPWORDS, ngram_range=(1, 2), min_df=1)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, good for short texts

# Embed each run's captions once and reuse across seeds (embeddings don't depend
# on the UMAP seed — only the clustering does), so N seeds don't re-embed N times.
_EMBEDDER = None


def embed(docs: list) -> np.ndarray:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDER.encode(docs, show_progress_bar=False)


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------

def build_topic_model(seed: int) -> BERTopic:
    # Seed UMAP so a given seed is reproducible. BERTopic's default UMAP is
    # stochastic; passing random_state pins it (this also forces n_jobs=1, so
    # it's slightly slower — fine for a corpus this size). These are BERTopic's
    # own UMAP defaults, with the seed added.
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric="cosine", random_state=seed)
    hdbscan_model = HDBSCAN(**HDBSCAN_KWARGS)
    vectorizer = CountVectorizer(**VECTORIZER_KWARGS)
    return BERTopic(
        embedding_model=EMBEDDING_MODEL,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
        nr_topics="auto",
    )


def run_bertopic(docs: list, image_ids: list, label: str, seed: int,
                 embeddings: np.ndarray, save_outputs: bool = False) -> tuple:
    """
    Fit BERTopic on docs for one seed. Returns (topic_assignments_df,
    topic_words_dict). When save_outputs is True (the reference seed), also
    writes per-image assignments and per-topic top words to data/topics/ for
    qualitative inspection — other seeds exist only to estimate variance.
    """
    model = build_topic_model(seed)
    topics, _ = model.fit_transform(docs, embeddings=embeddings)

    assignments = pd.DataFrame({
        "image_id": image_ids,
        "topic": topics,
    })

    # Top 10 words per topic (excluding noise topic -1)
    topic_words = {
        topic_id: [word for word, _ in words]
        for topic_id, words in model.get_topics().items()
        if topic_id != -1
    }

    if save_outputs:
        assignments.to_csv(TOPICS_DIR / f"{label}.csv", index=False)
        n_noise = (assignments["topic"] == -1).sum()
        print(f"  {label}: {len(topic_words)} topics, "
              f"{n_noise}/{len(docs)} noise docs (seed {seed})")
        words_df = pd.DataFrame([
            {"topic": tid,
             "size": int((assignments["topic"] == tid).sum()),
             "top_words": ", ".join(words[:10])}
            for tid, words in sorted(topic_words.items())
        ])
        words_df.to_csv(TOPICS_DIR / f"{label}_words.csv", index=False)

    return assignments, topic_words


# ---------------------------------------------------------------------------
# Coherence (CV)
# ---------------------------------------------------------------------------

def compute_coherence(topic_words: dict, tokenized_docs: list) -> float:
    """
    CV coherence using the captions as reference corpus.
    Returns NaN if fewer than 2 non-empty topics exist.
    """
    word_lists = [words for words in topic_words.values() if words]
    if len(word_lists) < 2:
        return float("nan")

    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    try:
        cm = CoherenceModel(
            topics=word_lists,
            texts=tokenized_docs,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_v",
        )
        return round(cm.get_coherence(), 4)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# NMI
# ---------------------------------------------------------------------------

def compute_nmi(assignments: pd.DataFrame, labels_df: pd.DataFrame) -> dict:
    """
    Merge topic assignments with ground-truth labels and compute NMI against
    each stratification dimension and against the *designed* experimental cell
    (the `cell` column, e.g. 'D_elites_high'), rather than a cell reconstructed
    from the three dimensions.

    Each NMI is computed on the rows that carry that particular label: some
    designed cells collapse a dimension (e.g. 'elites_low' has no `leaning`), so
    those rows still count toward nmi_cell even though they're dropped from
    nmi_leaning. Noise documents (topic == -1) are kept — they form their own
    implicit group. `n_cell` reports how many docs the cell NMI used.
    """
    merged = assignments.merge(labels_df, on="image_id", how="inner")
    results = {}
    for col, key in [("leaning", "nmi_leaning"),
                     ("actor_structure", "nmi_actor"),
                     ("salience", "nmi_salience"),
                     ("cell", "nmi_cell")]:
        if col not in merged.columns:
            results[key] = np.nan
            continue
        sub = merged.dropna(subset=[col])
        if sub.empty:
            results[key] = np.nan
            continue
        results[key] = round(
            normalized_mutual_info_score(
                sub["topic"].astype(str),
                sub[col].astype(str),
                average_method="arithmetic",
            ), 4
        )
    n_cell = merged["cell"].notna().sum() if "cell" in merged.columns else 0
    results["n_cell"] = int(n_cell)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all_captions() -> pd.DataFrame:
    dfs = []
    for path in CAPTIONS_DIR.glob("*.csv"):
        dfs.append(pd.read_csv(path))
    if not dfs:
        raise FileNotFoundError(f"No caption CSVs in {CAPTIONS_DIR}")
    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",  required=True,
                        help="CSV with image_id, leaning, actor_structure, salience")
    parser.add_argument("--refs",    required=True,
                        help="CSV with image_id, caption (NYT originals)")
    parser.add_argument("--output",  default=str(METRICS_DIR / "topic_results.csv"))
    parser.add_argument("--byseed-output",
                        default=str(METRICS_DIR / "topic_results_byseed.csv"),
                        help="long-form per-seed results (for plots / permutation tests)")
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS,
                        help="refit each run over this many seeds; report mean ± SD")
    args = parser.parse_args()

    labels_df = pd.read_csv(args.labels)
    labels_df["image_id"] = labels_df["image_id"].astype(str)

    refs_df = pd.read_csv(args.refs).dropna(subset=["caption"])
    refs_df["image_id"] = refs_df["image_id"].astype(str)

    captions_df = load_all_captions()
    captions_df["image_id"] = captions_df["image_id"].astype(str)

    # Build list of runs: (label, sub-dataframe with image_id + caption)
    runs = []

    # NYT reference captions as baseline
    runs.append(("nyt_reference", refs_df[["image_id", "caption"]]))

    # One run per model × condition
    for (model, condition), grp in captions_df.groupby(["model", "condition"]):
        runs.append((f"{model}_{condition}", grp[["image_id", "caption"]]))

    seeds = [RANDOM_SEED + i for i in range(args.n_seeds)]
    print(f"Robustness: {args.n_seeds} seed(s) per run -> {seeds}\n")

    # Metrics aggregated across seeds (mean ± SD).
    METRIC_KEYS = ["n_topics", "coherence_cv", "nmi_leaning",
                   "nmi_actor", "nmi_salience", "nmi_cell"]

    summary, byseed = [], []
    for label, run_df in tqdm(runs, desc="Topic modeling runs"):
        run_df = run_df.dropna(subset=["caption"])
        run_df = run_df[run_df["image_id"].isin(labels_df["image_id"])]
        if len(run_df) < 30:
            print(f"  Skipping {label}: too few documents ({len(run_df)})")
            continue

        docs = run_df["caption"].tolist()
        image_ids = run_df["image_id"].tolist()
        tokenized = [doc.lower().split() for doc in docs]
        embeddings = embed(docs)   # once per run, reused across seeds

        parts = label.split("_", 1)
        model_name = parts[0] if len(parts) == 2 else "nyt"
        condition = parts[1] if len(parts) == 2 else "reference"

        per_seed = {k: [] for k in METRIC_KEYS}
        n_cell = 0
        for si, seed in enumerate(seeds):
            assignments, topic_words = run_bertopic(
                docs, image_ids, label, seed, embeddings,
                save_outputs=(si == 0))   # reference seed writes qualitative files
            coherence = compute_coherence(topic_words, tokenized)
            nmi_scores = compute_nmi(assignments, labels_df)
            n_cell = nmi_scores["n_cell"]
            vals = {"n_topics": len(topic_words), "coherence_cv": coherence,
                    "nmi_leaning": nmi_scores["nmi_leaning"],
                    "nmi_actor": nmi_scores["nmi_actor"],
                    "nmi_salience": nmi_scores["nmi_salience"],
                    "nmi_cell": nmi_scores["nmi_cell"]}
            for k in METRIC_KEYS:
                per_seed[k].append(vals[k])
            byseed.append({"model": model_name, "condition": condition,
                           "seed": seed, "n_cell": n_cell, **vals})

        agg = {"model": model_name, "condition": condition,
               "n_docs": len(docs), "n_seeds": len(seeds), "n_cell": n_cell}
        for k in METRIC_KEYS:
            arr = np.array(per_seed[k], dtype=float)
            n_valid = int(np.sum(~np.isnan(arr)))
            agg[f"{k}_mean"] = round(float(np.nanmean(arr)), 4) if n_valid else np.nan
            agg[f"{k}_sd"] = round(float(np.nanstd(arr, ddof=1)), 4) if n_valid > 1 else 0.0
        summary.append(agg)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.output, index=False)
    byseed_df = pd.DataFrame(byseed)
    byseed_df.to_csv(args.byseed_output, index=False)

    print("\n" + "=" * 70)
    print(f"Topic modeling results — mean ± SD over {args.n_seeds} seeds")
    print("=" * 70)
    show = summary_df[["model", "condition", "n_topics_mean", "n_topics_sd",
                       "coherence_cv_mean", "nmi_cell_mean", "nmi_cell_sd"]]
    print(show.to_string(index=False))
    print(f"\nSummary (mean ± SD) saved to {args.output}")
    print(f"Per-seed long form saved to {args.byseed_output}")
    print("Reference-seed assignments + top words saved to data/topics/")


if __name__ == "__main__":
    main()
