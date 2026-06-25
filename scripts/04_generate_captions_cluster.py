"""
04_generate_captions_cluster.py

Generate captions for all images × conditions using the self-hosted open-source
VLMs (LLaVA-1.6, InternVL2, Qwen2.5-VL). Runs on a cluster GPU — see
cluster/run_vlm_inference.slurm (array job, one task per model).

This is the GPU counterpart to 04_generate_captions_api.py. It writes the SAME
CSV schema to the SAME directory, so 05_compute_metrics.py treats API and
cluster captions identically:
    data/captions/{model}_{condition}.csv
    Columns: image_id, model, condition, caption, prompt_tokens, completion_tokens, timestamp

Prompts and the four conditions are imported from caption_common, so they are
byte-identical to the API route.

Usage:
    python 04_generate_captions_cluster.py --model qwen
    python 04_generate_captions_cluster.py --model llava --conditions baseline textual_aux
    python 04_generate_captions_cluster.py --model all     # every local model, sequentially

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ IMPORTANT — each model handler follows that model's *official* inference  │
  │ recipe (HF model card), but the three families have very different        │
  │ multimodal APIs and CANNOT be verified without the weights + a GPU.       │
  │ Smoke-test each model on a handful of images (all four conditions)        │
  │ before launching the full corpus. See the pre-flight checklist.           │
  │                                                                           │
  │ Multi-image (visual_aux) support differs by family:                       │
  │   - Qwen2.5-VL : native multi-image — fine.                               │
  │   - InternVL2  : native multi-image via num_patches_list — fine.          │
  │   - LLaVA-1.6  : single-image model. Multiple images are concatenated     │
  │                  best-effort; treat visual_aux for LLaVA as a known weak   │
  │                  spot and inspect outputs.                                 │
  └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

# Reduce CUDA fragmentation (multi-image visual_aux is memory-spiky). Must be set
# before torch initializes CUDA — keep this above `import torch`.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from tqdm import tqdm

from config import CAPTIONS_DIR, CONDITIONS, IMAGES_DIR, LOCAL_MODELS, MODELS
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Per-condition inputs: (list of PIL images, prompt text)
# Identical decomposition to the API route's build_content().
# ---------------------------------------------------------------------------

def build_inputs(image_path: Path, condition: str, detection: dict):
    from PIL import Image

    main = Image.open(image_path).convert("RGB")

    if condition == "baseline":
        return [main], BASELINE_PROMPT

    if condition == "textual_aux":
        entity_list = format_entity_list(detection)
        return [main], TEXTUAL_AUX_PROMPT_TEMPLATE.format(entity_list=entity_list)

    if condition == "visual_aux":
        crops = crop_entities(image_path, detection)[:MAX_VISUAL_AUX_CROPS]
        return [main] + crops, VISUAL_AUX_PROMPT

    if condition == "annotated_aux":
        annotated = draw_annotations(image_path, detection)
        return [annotated], ANNOTATED_AUX_PROMPT

    raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Qwen2.5-VL
# ---------------------------------------------------------------------------

def load_qwen(model_id: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def generate_qwen(model, processor, images: list, prompt: str) -> dict:
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_CAPTION_TOKENS, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return {
        "caption": caption,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "completion_tokens": int(trimmed.shape[1]),
    }


# ---------------------------------------------------------------------------
# LLaVA-1.6 (LLaVA-NeXT)
# ---------------------------------------------------------------------------

def load_llava(model_id: str):
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
    ).eval()
    return model, processor


def generate_llava(model, processor, images: list, prompt: str) -> dict:
    # LLaVA-1.6 is single-image; for visual_aux we pass one <image> placeholder
    # per supplied image best-effort. Inspect these outputs (see header note).
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    conversation = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=text, return_tensors="pt").to(model.device, torch.float16)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_CAPTION_TOKENS, do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    caption = processor.decode(trimmed[0], skip_special_tokens=True).strip()
    return {
        "caption": caption,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "completion_tokens": int(trimmed.shape[1]),
    }


# ---------------------------------------------------------------------------
# InternVL2  (custom chat() interface + dynamic tiling preprocessing)
# Preprocessing adapted from the OpenGVLab/InternVL2 model card.
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_transform(input_size: int = 448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _find_closest_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff, best = float("inf"), (1, 1)
    area = width * height
    for r in target_ratios:
        tar = r[0] / r[1]
        diff = abs(aspect_ratio - tar)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]):
            best_diff, best = diff, r
    return best


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    w, h = image.size
    ar = w / h
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    rx, ry = _find_closest_ratio(ar, ratios, w, h, image_size)
    tw, th = image_size * rx, image_size * ry
    blocks = rx * ry
    resized = image.resize((tw, th))
    tiles = []
    cols = tw // image_size
    for i in range(blocks):
        box = ((i % cols) * image_size, (i // cols) * image_size,
               ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _internvl_pixel_values(image, input_size=448, max_num=12):
    transform = _internvl_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


def load_internvl(model_id: str):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def generate_internvl(model, tokenizer, images: list, prompt: str) -> dict:
    dtype = next(model.parameters()).dtype
    # Multi-image (visual_aux) explodes the tile count -> CUDA OOM: each image is
    # dynamically tiled into up to ~13 crops, so 6 images can become ~78 tiles.
    # Cap crops at 1 tile each; the main image keeps a normal (but smaller) budget.
    multi = len(images) > 1
    pv_list = []
    for i, img in enumerate(images):
        mx = 12 if not multi else (6 if i == 0 else 1)
        pv_list.append(_internvl_pixel_values(img, max_num=mx).to(dtype).to(model.device))
    num_patches_list = [pv.size(0) for pv in pv_list]
    pixel_values = torch.cat(pv_list, dim=0)

    # One "<image>" placeholder per supplied image (InternVL multi-image convention).
    if len(images) == 1:
        question = "<image>\n" + prompt
    else:
        prefix = "".join(f"Image-{i + 1}: <image>\n" for i in range(len(images)))
        question = prefix + prompt

    gen_config = dict(max_new_tokens=MAX_CAPTION_TOKENS, do_sample=False)
    caption = model.chat(
        tokenizer, pixel_values, question, gen_config,
        num_patches_list=num_patches_list, history=None, return_history=False,
    )
    del pixel_values, pv_list          # release tiles before the next image
    torch.cuda.empty_cache()
    return {"caption": caption.strip(), "prompt_tokens": None, "completion_tokens": None}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

HANDLERS = {
    "qwen":     (load_qwen, generate_qwen),
    "llava":    (load_llava, generate_llava),
    "internvl": (load_internvl, generate_internvl),
}


# ---------------------------------------------------------------------------
# Checkpointing / output  (same as the API route)
# ---------------------------------------------------------------------------

def load_checkpoint(model: str, condition: str) -> set:
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


# ---------------------------------------------------------------------------
# Run one model (single GPU, sequential — these are 7-8B models)
# ---------------------------------------------------------------------------

def run_model(model_name: str, conditions: list, image_ids: set = None,
              limit: int = None) -> None:
    if model_name not in HANDLERS:
        raise ValueError(f"No handler for '{model_name}'. Known: {list(HANDLERS)}")

    image_paths = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if image_ids is not None:
        image_paths = [p for p in image_paths if p.stem in image_ids]
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}"
              + ("" if image_ids is None else " matching the manifest"))
        return

    loader, generate = HANDLERS[model_name]
    model_id = MODELS[model_name]["model_id"]
    print(f"[{model_name}] loading {model_id} on {DEVICE} ...")
    model, processor = loader(model_id)
    print(f"[{model_name}] {len(image_paths)} images x {len(conditions)} conditions")

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
        for img_path in tqdm(remaining, desc=f"{model_name}/{condition}"):
            image_id = img_path.stem
            detection = load_detection(image_id)
            try:
                images, prompt = build_inputs(img_path, condition, detection)
                result = generate(model, processor, images, prompt)
                append_row(model_name, condition, {
                    "image_id": image_id,
                    "model": model_name,
                    "condition": condition,
                    "caption": result["caption"],
                    "prompt_tokens": result.get("prompt_tokens"),
                    "completion_tokens": result.get("completion_tokens"),
                    "timestamp": datetime.utcnow().isoformat(),
                })
                ok += 1
            except Exception as e:
                print(f"  ERROR {image_id} [{condition}]: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        total_ok += ok
        print(f"  {ok}/{len(remaining)} succeeded for {condition}")

    if total_attempted and total_ok == 0:
        raise RuntimeError(
            f"All {total_attempted} {model_name} caption calls FAILED (wrote 0 captions). "
            f"See the ERROR lines above — commonly a transformers/model-API mismatch, "
            f"CUDA OOM, or missing trust_remote_code. Smoke-test with --limit 2 first."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=LOCAL_MODELS + ["all"], default="all")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--references", default=None,
                        help="CSV with an image_id column; restrict captioning to these images")
    parser.add_argument("--limit", type=int, default=None,
                        help="Caption only the first N images (use 2-5 for a smoke test)")
    args = parser.parse_args()

    image_ids = None
    if args.references:
        ref = pd.read_csv(args.references)
        image_ids = set(ref["image_id"].astype(str))
        print(f"Restricting to {len(image_ids)} images from {args.references}")

    models = LOCAL_MODELS if args.model == "all" else [args.model]
    for model_name in models:
        run_model(model_name, args.conditions, image_ids, args.limit)


if __name__ == "__main__":
    main()
