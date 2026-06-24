"""
build_face_gallery.py

Bootstrap a face-recognition gallery from the NYT corpus WITHOUT hand-curating
reference images. The idea (weak supervision / label propagation):

  1. Embed every face in data/images/ with InsightFace.
  2. Cluster the embeddings (HDBSCAN). Each recurring politician forms a cluster.
  3. Label each cluster by majority vote over the `person_keywords` of the
     articles its faces came from. A cluster whose source images overwhelmingly
     tag "Trump, Donald J" *is* Trump — no manual labeling.
  4. For each confidently-named cluster, keep the centroid-nearest, quality-gated
     embeddings as gallery references.

Output: data/gallery/embeddings.pkl  →  dict[name] -> list[np.ndarray]
        (exactly the format load_gallery() / _match_embedding() expect in
         02_detect_entities.py, so detection needs no changes)

Why weak labels work despite multi-person photos: an image tagged with 3 people
contributes all 3 names, but across many images of the same face the *true*
identity recurs consistently while co-occurring names vary — so it wins the vote.

Usage:
    python build_face_gallery.py                 # full build (uses cached embeddings if present)
    python build_face_gallery.py --recompute     # re-embed all faces from scratch
    python build_face_gallery.py --report        # cluster/label diagnostics, write nothing
    python build_face_gallery.py --min-support 0.6 --min-margin 0.2 --max-refs 10

Tunables (all overridable on the CLI):
    --min-cluster-size   HDBSCAN min_cluster_size (smaller → more, smaller clusters)
    --min-support        a name must appear in >= this fraction of a cluster's
                         *labeled* source images to be assigned
    --min-margin         top name's support must beat the runner-up by this much
    --min-labeled        minimum # of labeled source images for a cluster to be named
    --max-refs           max reference embeddings stored per identity
"""

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import (
    DATA_DIR,
    FACE_QUALITY_THRESHOLD,
    GALLERY_DIR,
    IMAGES_DIR,
)

EMB_CACHE = GALLERY_DIR / "face_embeddings.npz"
ARCHIVE_CSV = DATA_DIR / "df_2025.csv"


# ──────────────────────────────────────────────────────────────────────────────
# image_id ↔ person_keywords mapping
# ──────────────────────────────────────────────────────────────────────────────

def clean_image_id(url: str) -> str:
    """Mirror of the notebook's clean_image_id so ids line up with data/images/."""
    stem = Path(str(url).split("?")[0]).stem
    return re.sub(
        r"-(jumbo|articleLarge|articleInline|master\d+|popup|"
        r"mediumThreeByTwo\d+|videoSmall|videoLarge|sfSpan|sub)$",
        "", stem,
    )


def normalize_name(raw: str) -> str:
    """'Trump, Donald J' / 'Johnson, Mike (1972- )' → 'Donald J Trump' / 'Mike Johnson'."""
    raw = re.sub(r"\([^)]*\)", "", raw).strip()          # drop '(1972- )' birth years
    if "," in raw:
        last, first = (p.strip() for p in raw.split(",", 1))
        name = f"{first} {last}".strip()
    else:
        name = raw
    return re.sub(r"\s+", " ", name).strip()


def build_image_to_names() -> dict[str, set[str]]:
    """image_id → set of normalized person names from df_2025.csv."""
    if not ARCHIVE_CSV.exists():
        raise FileNotFoundError(
            f"{ARCHIVE_CSV} not found — needed for weak labels (person_keywords)."
        )
    df = pd.read_csv(ARCHIVE_CSV, usecols=["multimedia_default_url", "person_keywords"])
    df = df.dropna(subset=["multimedia_default_url"])

    img2names: dict[str, set[str]] = {}
    for url, pk in zip(df["multimedia_default_url"], df["person_keywords"]):
        if not str(url).startswith("http"):
            continue
        iid = clean_image_id(url)
        names = set()
        if isinstance(pk, str) and pk.strip() and pk.strip().lower() != "nan":
            names = {normalize_name(n) for n in pk.split(";") if n.strip()}
            names = {n for n in names if n}
        # An image can recur across rows; union the tagged names.
        img2names.setdefault(iid, set()).update(names)
    return img2names


# ──────────────────────────────────────────────────────────────────────────────
# Face embedding (cached)
# ──────────────────────────────────────────────────────────────────────────────

def load_face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0)
    return app


def embed_corpus_faces(recompute: bool = False):
    """Return (embeddings [N,512] float32 L2-normalized, image_ids [N], det_scores [N])."""
    if EMB_CACHE.exists() and not recompute:
        d = np.load(EMB_CACHE, allow_pickle=True)
        print(f"Loaded cached embeddings: {len(d['image_ids'])} faces from {EMB_CACHE.name}")
        return d["embeddings"], d["image_ids"], d["det_scores"]

    app = load_face_app()
    image_paths = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images in {IMAGES_DIR}")

    embs, ids, scores = [], [], []
    for p in tqdm(image_paths, desc="Embedding faces"):
        try:
            img_bgr = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        for face in app.get(img_bgr):
            emb = face.embedding.astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)   # L2-normalize for cosine
            embs.append(emb)
            ids.append(p.stem)
            scores.append(float(face.det_score))

    embeddings = np.vstack(embs).astype(np.float32)
    image_ids = np.array(ids)
    det_scores = np.array(scores, dtype=np.float32)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB_CACHE, embeddings=embeddings,
                        image_ids=image_ids, det_scores=det_scores)
    print(f"Embedded {len(image_ids)} faces from {len(image_paths)} images → {EMB_CACHE.name}")
    return embeddings, image_ids, det_scores


# ──────────────────────────────────────────────────────────────────────────────
# Cluster + label-propagate
# ──────────────────────────────────────────────────────────────────────────────

def cluster_faces(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """HDBSCAN on L2-normalized embeddings (euclidean ∝ cosine). Falls back to DBSCAN."""
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
        print(f"HDBSCAN: {labels.max() + 1} clusters, "
              f"{(labels == -1).sum()} noise / {len(labels)} faces")
    except ImportError:
        from sklearn.cluster import DBSCAN
        # cosine sim 0.4 ≈ euclidean 1.095 on unit vectors: d = sqrt(2-2cos)
        labels = DBSCAN(eps=1.0, min_samples=min_cluster_size,
                        metric="euclidean").fit_predict(embeddings)
        print(f"DBSCAN (hdbscan unavailable): {labels.max() + 1} clusters, "
              f"{(labels == -1).sum()} noise / {len(labels)} faces")
    return labels


def assign_names(labels, image_ids, img2names, *,
                 min_support, min_margin, min_labeled):
    """
    For each cluster, vote names over its *labeled* source images.

    Returns {cluster_id: {"name", "support", "margin", "n_imgs", "n_labeled"}}
    for clusters that pass the confidence gates, plus a list of all-cluster
    diagnostics for reporting.
    """
    assignments, diagnostics = {}, []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        cluster_imgs = {image_ids[i] for i in idx}
        labeled_imgs = [iid for iid in cluster_imgs if img2names.get(iid)]
        n_imgs, n_labeled = len(cluster_imgs), len(labeled_imgs)

        # support = fraction of LABELED images whose name set contains the candidate
        votes = Counter()
        for iid in labeled_imgs:
            for name in img2names[iid]:
                votes[name] += 1
        ranked = [(n, c / n_labeled) for n, c in votes.most_common()] if n_labeled else []

        top_name, top_sup = (ranked[0] if ranked else (None, 0.0))
        second_sup = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_sup - second_sup
        passes = (n_labeled >= min_labeled and top_sup >= min_support
                  and margin >= min_margin)

        diagnostics.append({
            "cluster": cid, "n_imgs": n_imgs, "n_labeled": n_labeled,
            "top_name": top_name, "support": round(top_sup, 2),
            "margin": round(margin, 2), "assigned": passes,
        })
        if passes:
            assignments[cid] = {"name": top_name, "support": top_sup,
                                "margin": margin, "n_imgs": n_imgs,
                                "n_labeled": n_labeled}
    return assignments, diagnostics


def select_references(idx, embeddings, det_scores, max_refs):
    """Quality-gate, then take the centroid-nearest faces as clean references."""
    keep = [i for i in idx if det_scores[i] >= FACE_QUALITY_THRESHOLD]
    if not keep:
        keep = list(idx)  # fall back to all if everything is below the gate
    sub = embeddings[keep]
    centroid = sub.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-8)
    order = np.argsort(-(sub @ centroid))           # cosine to centroid, descending
    chosen = [keep[j] for j in order[:max_refs]]
    return [embeddings[i] for i in chosen]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap a face gallery from the NYT corpus.")
    ap.add_argument("--recompute", action="store_true", help="Re-embed all faces (ignore cache)")
    ap.add_argument("--report", action="store_true", help="Print diagnostics, write nothing")
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--min-support", type=float, default=0.6)
    ap.add_argument("--min-margin", type=float, default=0.2)
    ap.add_argument("--min-labeled", type=int, default=3)
    ap.add_argument("--max-refs", type=int, default=10)
    args = ap.parse_args()

    img2names = build_image_to_names()
    embeddings, image_ids, det_scores = embed_corpus_faces(recompute=args.recompute)
    labels = cluster_faces(embeddings, args.min_cluster_size)
    assignments, diagnostics = assign_names(
        labels, image_ids, img2names,
        min_support=args.min_support, min_margin=args.min_margin,
        min_labeled=args.min_labeled,
    )

    print(f"\n{len(assignments)} clusters confidently named "
          f"(of {labels.max() + 1} total).")

    if args.report:
        print("\ncluster  n_imgs  n_lab  support  margin  assigned  name")
        for d in sorted(diagnostics, key=lambda x: -x["n_imgs"]):
            print(f"{d['cluster']:>7}  {d['n_imgs']:>6}  {d['n_labeled']:>5}  "
                  f"{d['support']:>7}  {d['margin']:>6}  {str(d['assigned']):>8}  "
                  f"{d['top_name']}")
        return

    # Merge clusters that resolve to the same name (e.g. different poses/eras).
    gallery: dict[str, list[np.ndarray]] = {}
    for cid, info in assignments.items():
        idx = np.where(labels == cid)[0]
        refs = select_references(idx, embeddings, det_scores, args.max_refs)
        gallery.setdefault(info["name"], []).extend(refs)

    # Cap per-identity refs after merging.
    for name in gallery:
        gallery[name] = gallery[name][: args.max_refs]

    out = GALLERY_DIR / "embeddings.pkl"
    with open(out, "wb") as f:
        pickle.dump(gallery, f)
    print(f"\nGallery saved → {out}  ({len(gallery)} identities)")
    print("Top identities by reference count:")
    for name, refs in sorted(gallery.items(), key=lambda x: -len(x[1]))[:20]:
        print(f"  {len(refs):>3}  {name}")


if __name__ == "__main__":
    main()
