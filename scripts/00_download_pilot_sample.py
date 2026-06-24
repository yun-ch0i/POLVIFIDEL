"""
00_download_pilot_sample.py

Download a small pilot sample of NYT images from df_2025.csv for end-to-end testing.

Usage:
    python 00_download_pilot_sample.py --n 20
    python 00_download_pilot_sample.py --n 20 --seed 42

Outputs:
    data/images/{image_id}.jpg    ← downloaded images
    data/df_pilot.csv             ← image_id + reference caption for 05_compute_metrics.py
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import DATA_DIR, IMAGES_DIR

PILOT_CSV = DATA_DIR / "df_pilot.csv"

# Sections most likely to contain politically salient images
POLITICAL_DESKS = {
    "Politics", "National", "Washington", "Foreign", "U.S.",
    "World", "Business", "Opinion",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (research bot; dissertation project)"}


def clean_image_id(url: str) -> str:
    """Derive a stable image_id from the NYT image URL."""
    stem = Path(url.split("?")[0]).stem  # strip query params, get filename stem
    # Remove NYT size suffixes like -mediumThreeByTwo210, -jumbo, -articleLarge
    stem = re.sub(
        r"-(jumbo|articleLarge|articleInline|master\d+|popup|"
        r"mediumThreeByTwo\d+|videoSmall|videoLarge|sfSpan|sub)$",
        "",
        stem,
    )
    return stem


def download_image(url: str, dest: Path) -> bool:
    """Download image to dest. Returns True on success."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        dest.write_bytes(resp.content)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of images to download")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--csv", default=str(DATA_DIR / "df_2025.csv"),
        help="Path to the full NYT archive CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Keep rows with both image URL and caption
    df = df.dropna(subset=["multimedia_default_url", "caption"])
    df = df[df["multimedia_default_url"].str.startswith("http")]
    print(f"  {len(df)} rows with image URL + caption")

    # Prefer politically relevant desks; fall back to all if not enough
    political = df[df["news_desk"].isin(POLITICAL_DESKS)]
    pool = political if len(political) >= args.n * 3 else df
    print(f"  Sampling from {len(pool)} rows "
          f"({'political desks' if pool is political else 'full archive'})")

    # Deduplicate: one image per article web_url
    if "web_url" in pool.columns:
        pool = pool.drop_duplicates(subset=["web_url"])

    pool = pool.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Download until we have --n successes
    rows = []
    already_done = {p.stem for p in IMAGES_DIR.glob("*.jpg")}

    pbar = tqdm(pool.iterrows(), total=len(pool), desc="Downloading")
    for _, record in pbar:
        if len(rows) >= args.n:
            break
        url = record["multimedia_default_url"]
        image_id = clean_image_id(url)

        if image_id in already_done:
            rows.append({"image_id": image_id, "caption": record["caption"]})
            pbar.set_postfix(downloaded=len(rows), skipped="(exists)")
            continue

        dest = IMAGES_DIR / f"{image_id}.jpg"
        if download_image(url, dest):
            rows.append({"image_id": image_id, "caption": record["caption"]})
            already_done.add(image_id)
            pbar.set_postfix(downloaded=len(rows))
        else:
            pbar.set_postfix(downloaded=len(rows), last="failed")

        time.sleep(0.2)  # polite crawl delay

    df_pilot = pd.DataFrame(rows)
    df_pilot.to_csv(PILOT_CSV, index=False)
    print(f"\nDownloaded {len(df_pilot)} images → {IMAGES_DIR}")
    print(f"Pilot metadata saved → {PILOT_CSV}")


if __name__ == "__main__":
    main()
