"""
04_generate_captions_api.py

Generate captions for all images × conditions using the VLMs served by
Princeton AI Sandbox (GPT-4o, Gemini, Claude family, Llama-4-Scout, Mistral).
Runs locally / on a login node — no GPU required, no per-call billing.

Every Sandbox model is reached through ONE OpenAI-compatible client; the only
thing that varies per model is the `model_id` string from config.MODELS.
Self-hosted open-source VLMs (llava/internvl/qwen) go through the GPU route in
04_generate_captions_cluster.py instead.

Usage:
    python 04_generate_captions_api.py --model gpt4o
    python 04_generate_captions_api.py --model claude-sonnet
    python 04_generate_captions_api.py --model all        # every api model

Outputs:
    data/captions/{model}_{condition}.csv
    Columns: image_id, model, condition, caption, prompt_tokens, completion_tokens, timestamp

Checkpointing: already-completed (image, condition) pairs are skipped on re-run.

Credentials:
    export AI_SANDBOX_KEY=...        # Princeton AI Sandbox key (Portkey gateway)
"""

from __future__ import annotations

import argparse
import base64
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import (
    AI_SANDBOX_KEY,
    API_MODELS,
    CAPTIONS_DIR,
    CONDITIONS,
    IMAGES_DIR,
    MODELS,
)
from caption_common import (
    ANNOTATED_AUX_PROMPT,
    BASELINE_PROMPT,
    MAX_CAPTION_TOKENS,
    MAX_VISUAL_AUX_CROPS,
    TEXTUAL_AUX_PROMPT_TEMPLATE,
    VISUAL_AUX_PROMPT,
    crop_entities,
    draw_annotations,
    format_entity_list,
    load_detection,
    matched_entities,
)

# ---------------------------------------------------------------------------
# Route-specific helpers (base64 encoding for the OpenAI-style payload)
# Prompts and detection helpers are imported from caption_common.
# ---------------------------------------------------------------------------

def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# AI Sandbox client (one OpenAI-compatible client for every api model)
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()
_sandbox_client = None


def get_sandbox_client():
    """Build the shared Portkey client for Princeton AI Sandbox.

    The gateway is OpenAI-compatible: client.chat.completions.create(...) works
    identically, so the rest of this module is unchanged. Auth is the single
    AI_SANDBOX_KEY; no base_url / api_version needed.
    """
    global _sandbox_client
    if _sandbox_client is None:
        with _write_lock:
            if _sandbox_client is None:
                from portkey_ai import Portkey
                _sandbox_client = Portkey(api_key=AI_SANDBOX_KEY)
    return _sandbox_client


def build_content(image_path: Path, condition: str, detection: dict) -> list:
    """Assemble the OpenAI-style multimodal content list for a condition.

    Identical across every Sandbox model, since they all share the chat
    completions image_url format.
    """
    content = []

    if condition == "baseline":
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
        content.append({"type": "text", "text": BASELINE_PROMPT})

    elif condition == "textual_aux":
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
        entity_list = format_entity_list(detection)
        content.append({"type": "text",
                         "text": TEXTUAL_AUX_PROMPT_TEMPLATE.format(entity_list=entity_list)})

    elif condition == "visual_aux":
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
        crops = crop_entities(image_path, detection)
        for crop in crops[:MAX_VISUAL_AUX_CROPS]:  # cap crops to stay within token limits
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{pil_to_b64(crop)}"}})
        content.append({"type": "text", "text": VISUAL_AUX_PROMPT})

    elif condition == "annotated_aux":
        annotated = draw_annotations(image_path, detection)
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{pil_to_b64(annotated)}"}})
        content.append({"type": "text", "text": ANNOTATED_AUX_PROMPT})

    return content


def call_sandbox(model_name: str, image_path: Path, condition: str, detection: dict) -> dict:
    client = get_sandbox_client()
    content = build_content(image_path, condition, detection)
    response = client.chat.completions.create(
        model=MODELS[model_name]["model_id"],
        messages=[{"role": "user", "content": content}],
        max_tokens=MAX_CAPTION_TOKENS,
        temperature=0,   # deterministic / reproducible captions (matches the greedy cluster route)
    )
    usage = response.usage
    return {
        "caption": response.choices[0].message.content.strip(),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def load_checkpoint(model: str, condition: str) -> set[str]:
    out_path = CAPTIONS_DIR / f"{model}_{condition}.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        return set(df["image_id"].astype(str))
    return set()


def append_row(model: str, condition: str, row: dict) -> None:
    out_path = CAPTIONS_DIR / f"{model}_{condition}.csv"
    df_new = pd.DataFrame([row])
    if out_path.exists():
        df_new.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(out_path, index=False)


def run_model(model_name: str, conditions: list[str], image_ids: set[str] | None = None,
              limit: int | None = None, workers: int = 8) -> None:
    if not AI_SANDBOX_KEY:
        print(f"Skipping {model_name}: AI_SANDBOX_KEY not set")
        return

    image_paths = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if image_ids is not None:
        image_paths = [p for p in image_paths if p.stem in image_ids]
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}"
              + ("" if image_ids is None else " matching the manifest"))
        return
    print(f"[{model_name}] {len(image_paths)} images x {len(conditions)} conditions"
          + (f" (--limit {limit})" if limit else "") + f"  [{workers} parallel workers]")

    def caption_one(img_path: Path, condition: str) -> bool:
        image_id = img_path.stem
        detection = load_detection(image_id)
        for attempt in range(4):
            try:
                result = call_sandbox(model_name, img_path, condition, detection)
                row = {
                    "image_id": image_id,
                    "model": model_name,
                    "condition": condition,
                    "caption": result["caption"],
                    "prompt_tokens": result.get("prompt_tokens"),
                    "completion_tokens": result.get("completion_tokens"),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                with _write_lock:
                    append_row(model_name, condition, row)
                return True
            except Exception as e:
                if attempt == 3:
                    print(f"  ERROR {image_id}: {e}")
                    return False
                time.sleep(2 ** attempt)   # backoff on rate-limit / transient error

    total_attempted = total_ok = 0
    for condition in conditions:
        done = load_checkpoint(model_name, condition)
        remaining = [p for p in image_paths if p.stem not in done]
        if condition != "baseline":   # aux only helps where there's labeled grounding
            before = len(remaining)
            remaining = [p for p in remaining if matched_entities(load_detection(p.stem))]
            n_skip = before - len(remaining)
            if n_skip:
                print(f"[{model_name} / {condition}] skipping {n_skip} images with no "
                      f"matched-labeled entity (aux == baseline there; saves tokens)")
        print(f"[{model_name} / {condition}] {len(done)} done, {len(remaining)} remaining")
        if not remaining:
            continue
        total_attempted += len(remaining)
        ok = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(caption_one, p, condition) for p in remaining]
            for fut in tqdm(as_completed(futures), total=len(remaining),
                            desc=f"{model_name}/{condition}"):
                if fut.result():
                    ok += 1
        total_ok += ok
        print(f"  {ok}/{len(remaining)} succeeded for {condition}")

    if total_attempted and total_ok == 0:
        raise RuntimeError(
            f"All {total_attempted} {model_name} caption calls FAILED (wrote 0 captions). "
            f"See the ERROR lines above for the cause (commonly an invalid AI_SANDBOX_KEY, "
            f"an unrecognized model_id, or a rate limit). Fix that, then re-run."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=API_MODELS + ["all"], default="all")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--references", default=None,
                        help="CSV with an image_id column; restrict captioning to these images")
    parser.add_argument("--limit", type=int, default=None,
                        help="Caption only the first N images (for a quick floor-effect check)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API requests (lower if you hit rate limits, raise on higher tiers)")
    args = parser.parse_args()

    image_ids = None
    if args.references:
        ref = pd.read_csv(args.references)
        image_ids = set(ref["image_id"].astype(str))
        print(f"Restricting to {len(image_ids)} images from {args.references}")

    models = API_MODELS if args.model == "all" else [args.model]
    for model_name in models:
        run_model(model_name, args.conditions, image_ids, args.limit, args.workers)


if __name__ == "__main__":
    main()
