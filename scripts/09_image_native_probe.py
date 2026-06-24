"""
09_image_native_probe.py

RQ4 — when to go image-native vs. text (caption) for downstream political analysis.

Measures how much political signal is *linearly recoverable* from two
representations of the same images, using one shared probe:

  1. IMAGE-NATIVE  — CLIP image embeddings (no captioning step).
  2. CAPTION-TEXT  — TF-IDF over each model×condition's captions (and the NYT
                     human captions as the ceiling).

The probe is a 5-fold cross-validated logistic regression predicting the
political label (the designed `cell`, and `leaning`), scored by macro-F1 against
a most-frequent-class baseline. Comparing image-native vs. caption F1 on the
*same images* quantifies the information lost by the image -> text round-trip.

Usage:
    python 09_image_native_probe.py --references data/df_corpus.csv

Outputs:
    data/metrics/native_vs_caption_probe.csv  — one row per representation×target
    data/metrics/image_clip_embeddings.npz    — cached CLIP embeddings (reused)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import normalize

from config import CAPTIONS_DIR, IMAGES_DIR, METRICS_DIR, OBJECT_GALLERY_CLIP

warnings.filterwarnings("ignore")

EMB_CACHE = METRICS_DIR / "image_clip_embeddings.npz"
TARGETS = ["cell", "leaning"]
MIN_PER_CLASS = 5     # classes with fewer members can't survive 5-fold CV
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# CLIP image embeddings (cached)
# ---------------------------------------------------------------------------

def embed_images(image_ids: list) -> dict:
    """Return {image_id: np.ndarray} CLIP image embeddings, caching to disk."""
    cache = {}
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=True)
        cache = {i: v for i, v in zip(z["ids"], z["emb"])}
        print(f"Loaded {len(cache)} cached embeddings from {EMB_CACHE.name}")

    todo = [i for i in image_ids if i not in cache]
    if todo:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(OBJECT_GALLERY_CLIP).to(device).eval()
        proc = CLIPProcessor.from_pretrained(OBJECT_GALLERY_CLIP)
        print(f"Embedding {len(todo)} images with CLIP ({OBJECT_GALLERY_CLIP}) on {device}")

        def find(iid):
            for ext in (".jpg", ".png", ".jpeg"):
                p = IMAGES_DIR / f"{iid}{ext}"
                if p.exists():
                    return p
            return None

        batch, batch_ids = [], []

        def flush():
            if not batch:
                return
            inp = proc(images=batch, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.get_image_features(**inp)
            # This laion checkpoint can return a model-output object rather than a
            # tensor (same as embed_crop in 02_detect_entities) — extract the embedding.
            if torch.is_tensor(out):
                t = out
            elif getattr(out, "image_embeds", None) is not None:
                t = out.image_embeds
            elif getattr(out, "pooler_output", None) is not None:
                t = out.pooler_output
            else:
                t = out.last_hidden_state.mean(dim=1)
            feats = t.detach().cpu().numpy()
            for bid, f in zip(batch_ids, feats):
                cache[bid] = f
            batch.clear(); batch_ids.clear()

        for n, iid in enumerate(todo, 1):
            p = find(iid)
            if p is None:
                continue
            try:
                batch.append(Image.open(p).convert("RGB"))
                batch_ids.append(iid)
            except Exception:
                continue
            if len(batch) >= 32:
                flush()
            if n % 200 == 0:
                print(f"  embedded {n}/{len(todo)}")
        flush()

        ids = np.array(list(cache.keys()))
        emb = np.stack([cache[i] for i in ids])
        np.savez(EMB_CACHE, ids=ids, emb=emb)
        print(f"Cached {len(ids)} embeddings -> {EMB_CACHE.name}")

    return cache


# ---------------------------------------------------------------------------
# Shared probe
# ---------------------------------------------------------------------------

def probe(X, y) -> tuple:
    """5-fold CV logistic regression; returns (n, macro_f1, baseline_macro_f1).

    X may be dense (image embeddings) or sparse (TF-IDF). Classes with fewer
    than MIN_PER_CLASS members are dropped so stratified CV is valid.
    """
    y = np.asarray(y)
    keep_classes = [c for c, n in zip(*np.unique(y, return_counts=True)) if n >= MIN_PER_CLASS]
    mask = np.isin(y, keep_classes)
    X, y = X[mask], y[mask]
    if len(np.unique(y)) < 2 or X.shape[0] < 50:
        return None
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pred = cross_val_predict(clf, X, y, cv=cv)
    base = cross_val_predict(DummyClassifier(strategy="most_frequent"), X, y, cv=cv)
    return X.shape[0], f1_score(y, pred, average="macro"), f1_score(y, base, average="macro")


def record(rows, representation, target, res):
    if res is None:
        return
    n, f1, base = res
    rows.append({"representation": representation, "target": target, "n": n,
                 "f1_macro": round(f1, 4), "baseline_f1": round(base, 4),
                 "lift": round(f1 - base, 4)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--references", required=True,
                    help="CSV with image_id, caption, cell, leaning (df_corpus.csv)")
    ap.add_argument("--output", default=str(METRICS_DIR / "native_vs_caption_probe.csv"))
    ap.add_argument("--limit", type=int, default=None, help="smoke test: cap #images embedded")
    args = ap.parse_args()

    labels = pd.read_csv(args.references)
    labels["image_id"] = labels["image_id"].astype(str)
    if args.limit:
        labels = labels.head(args.limit)

    # CLIP embeddings for every corpus image
    emb = embed_images(labels["image_id"].tolist())
    have = labels["image_id"].isin(emb.keys())
    print(f"{have.sum()}/{len(labels)} corpus images have embeddings")

    rows = []

    # --- 1. IMAGE-NATIVE probe (all labeled images) ---
    for target in TARGETS:
        d = labels[have].dropna(subset=[target])
        if d.empty:
            continue
        X = normalize(np.stack([emb[i] for i in d["image_id"]]))   # L2-normalize CLIP
        record(rows, "image_clip", target, probe(X, d[target].values))

    # --- 2. CAPTION-TEXT probes + matched image-native deltas ---
    sources = [("caption:nyt_reference", labels[["image_id", "caption"]])]
    for p in sorted(CAPTIONS_DIR.glob("*.csv")):
        c = pd.read_csv(p); c["image_id"] = c["image_id"].astype(str)
        sources.append((f"caption:{p.stem}", c[["image_id", "caption"]]))

    for name, cap in sources:
        cap = cap.dropna(subset=["caption"])
        merged = cap.merge(labels[["image_id"] + TARGETS], on="image_id", how="inner")
        for target in TARGETS:
            d = merged.dropna(subset=[target])
            if len(d) < 50:
                continue
            Xc = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=3, max_features=5000).fit_transform(d["caption"])
            record(rows, name, target, probe(Xc, d[target].values))
            # matched image-native baseline on the SAME images (only for VLM sources)
            if name != "caption:nyt_reference":
                di = d[d["image_id"].isin(emb.keys())]
                if len(di) >= 50:
                    Xi = normalize(np.stack([emb[i] for i in di["image_id"]]))
                    record(rows, f"image_clip@{name.split(':')[1]}", target,
                           probe(Xi, di[target].values))

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    print("\n" + "=" * 72)
    print("Recoverable political signal — image-native vs. caption (macro-F1)")
    print("=" * 72)
    for target in TARGETS:
        sub = out[out["target"] == target].sort_values("f1_macro", ascending=False)
        print(f"\n--- target: {target} (baseline ~{sub['baseline_f1'].median():.3f}) ---")
        print(sub[["representation", "n", "f1_macro", "lift"]].to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
