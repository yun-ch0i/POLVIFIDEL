"""
download_corpus.py

Download the images listed in a corpus manifest (default: data/df_corpus.csv,
produced by 01b_consolidate_corpus.py) into data/images/.

Idempotent: images already present are skipped, so this can be re-run after a
top-up of the manifest without re-fetching anything.

Usage:
    python download_corpus.py
    python download_corpus.py --manifest data/df_corpus.csv --sleep 0.2
    python download_corpus.py --retry-failed       # only re-attempt previously failed ids

Inputs:
    --manifest CSV   - must have columns: image_id, multimedia_default_url

Outputs:
    data/images/{image_id}.jpg
    data/images/_download_failures.csv   - rows that failed (for inspection / retry)
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
import requests
from tqdm import tqdm

from config import DATA_DIR, IMAGES_DIR

FAILURES_CSV = IMAGES_DIR / "_download_failures.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def download_image(url: str, dest, timeout: int) -> tuple[bool, str]:
    """Fetch one image. Returns (ok, reason)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        if "image" not in r.headers.get("content-type", ""):
            return False, "not-an-image"
        dest.write_bytes(r.content)
        return True, "ok"
    except Exception as e:  # noqa: BLE001 - report any failure, keep going
        return False, type(e).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DATA_DIR / "df_corpus.csv"))
    parser.add_argument("--sleep", type=float, default=0.2,
                        help="Seconds to pause between downloads (be polite)")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only attempt ids recorded in _download_failures.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest).dropna(subset=["multimedia_default_url"])
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset="image_id")

    if args.retry_failed:
        if not FAILURES_CSV.exists():
            print("No failures file — nothing to retry.")
            return
        retry_ids = set(pd.read_csv(FAILURES_CSV)["image_id"].astype(str))
        df = df[df["image_id"].isin(retry_ids)]

    already = {p.stem for p in IMAGES_DIR.glob("*.jpg")}
    todo = df[~df["image_id"].isin(already)]
    print(f"{len(already)} already on disk, {len(todo)} to download "
          f"(of {len(df)} in manifest)")

    ok, failures = 0, []
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="Downloading"):
        iid = row["image_id"]
        dest = IMAGES_DIR / f"{iid}.jpg"
        success, reason = download_image(row["multimedia_default_url"], dest,
                                         args.timeout)
        if success:
            ok += 1
        else:
            failures.append({"image_id": iid,
                             "multimedia_default_url": row["multimedia_default_url"],
                             "reason": reason})
        time.sleep(args.sleep)

    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_CSV, index=False)
    elif FAILURES_CSV.exists() and args.retry_failed:
        FAILURES_CSV.unlink()  # all retries succeeded

    print(f"\nDownloaded {ok} new images. {len(failures)} failed.")
    total = len({p.stem for p in IMAGES_DIR.glob('*.jpg')})
    print(f"images/ now holds {total} images.")
    if failures:
        print(f"Failures written to {FAILURES_CSV} — re-run with --retry-failed.")


if __name__ == "__main__":
    main()
