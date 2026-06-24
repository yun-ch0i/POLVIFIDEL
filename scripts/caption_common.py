"""
caption_common.py

Shared caption-generation logic used by BOTH routes:
  - 04_generate_captions_api.py     (Sandbox / API models, no GPU)
  - 04_generate_captions_cluster.py (self-hosted open-source VLMs, GPU)

Everything that defines the *experiment* lives here — the prompts and the four
prompting conditions — so the two routes stay byte-identical and the resulting
captions are comparable across all models. Route-specific plumbing (base64
encoding for the API, HF model loading for the cluster) stays in each script.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import DETECTIONS_DIR

# ---------------------------------------------------------------------------
# Prompts  (identical across every model and both routes)
# ---------------------------------------------------------------------------

# Target caption length enforced both in the prompt and via max_tokens.
# "Up to 3 sentences" (rather than "exactly 3") prevents padding/hallucination
# on simpler images while giving enough text for topic modeling and frame analysis.
CAPTION_LENGTH_INSTRUCTION = (
    "Write up to 3 sentences, no more than 100 words in total."
)
MAX_CAPTION_TOKENS = 160  # ~100 words + buffer

BASELINE_PROMPT = (
    "Describe this image. Focus on the people present, their identities, "
    "any objects or symbols with political significance, and the "
    "relationships or interactions between people. Be specific and factual. "
    + CAPTION_LENGTH_INSTRUCTION
)

TEXTUAL_AUX_PROMPT_TEMPLATE = (
    "The following political entities were automatically detected in this image:\n"
    "{entity_list}\n\n"
    "Using this information as grounding, describe the image. Focus on "
    "the people present, their identities, any objects or symbols with political "
    "significance, and the relationships or interactions between people. "
    "Be specific and factual. " + CAPTION_LENGTH_INSTRUCTION
)

VISUAL_AUX_PROMPT = (
    "The cropped images provided show political entities detected in the main image. "
    "Using these as grounding, describe the main image. Focus on the people "
    "present, their identities, any objects or symbols with political significance, "
    "and the relationships or interactions between people. Be specific and factual. "
    + CAPTION_LENGTH_INSTRUCTION
)

ANNOTATED_AUX_PROMPT = (
    "The image has bounding boxes and labels marking detected political entities. "
    "Using these annotations as grounding, describe the image. Focus on "
    "the people present, their identities, any objects or symbols with political "
    "significance, and the relationships or interactions between people. "
    "Be specific and factual. " + CAPTION_LENGTH_INSTRUCTION
)

# Cap on auxiliary crops fed to the visual_aux condition, shared so both routes
# truncate the same way.
MAX_VISUAL_AUX_CROPS = 5


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def load_detection(image_id: str) -> dict:
    det_path = DETECTIONS_DIR / f"{image_id}.json"
    if not det_path.exists():
        return {"objects": [], "faces": []}
    with open(det_path) as f:
        return json.load(f)


def matched_entities(detection: dict) -> list[dict]:
    """Entities matched to a LABELED identity/subcategory — the ONLY grounding the
    aux conditions are allowed to use. Excludes 'unknown'/low-quality faces and any
    object not matched to a labeled gallery subcategory.

    Returns [{"bbox", "label", "kind"}], label = person name or object subcategory
    (snake_case → spaces). All three aux conditions draw from this same set, so they
    differ only in modality (text / crops / annotation), never in coverage.
    """
    items = []
    for f in detection.get("faces", []):
        if f.get("match_status") == "matched" and f.get("name"):
            items.append({"bbox": f.get("bbox"), "label": f["name"], "kind": "face",
                          "score": float(f.get("det_score") or 0.0)})
    for o in detection.get("objects", []):
        if o.get("match_status") == "matched" and o.get("subcategory"):
            items.append({"bbox": o.get("bbox"),
                          "label": str(o["subcategory"]).replace("_", " "),
                          "kind": "object",
                          "score": float(o.get("score") or 0.0)})
    return items


def format_entity_list(detection: dict) -> str:
    items = matched_entities(detection)
    people = [it["label"] for it in items if it["kind"] == "face"]
    objects = [it["label"] for it in items if it["kind"] == "object"]
    lines = []
    if people:
        lines.append("Identified people: " + ", ".join(sorted(set(people))))
    if objects:
        counts = Counter(objects)
        lines.append("Detected objects: " + ", ".join(
            f"{n} {label}" if n > 1 else label
            for label, n in counts.most_common()
        ))
    return "\n".join(lines) if lines else "No political entities detected."


def _load_font(size: int):
    """Find a TrueType bold font across OSes; fall back to PIL's default at `size`."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",          # Debian/Ubuntu/Colab
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",            # macOS
        "DejaVuSans-Bold.ttf",                                          # PIL-bundled, if present
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10 supports size
    except TypeError:
        return ImageFont.load_default()


def draw_annotations(image_path: Path, detection: dict) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, _ = img.size
    font = _load_font(max(14, W // 45))   # scale label size to the image
    box_w = max(2, W // 300)
    for it in matched_entities(detection):
        bbox = it.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1, y1], outline="red", width=box_w)
        label = it["label"]
        try:
            l, t, r, b = draw.textbbox((0, 0), label, font=font)
            tw, th = r - l, b - t
        except Exception:                       # very old Pillow
            tw, th = draw.textsize(label, font=font)
        pad = 2
        ty = max(0, y0 - th - 2 * pad)          # caption sits above the box, white-on-red
        draw.rectangle([x0, ty, x0 + tw + 2 * pad, ty + th + 2 * pad], fill="red")
        draw.text((x0 + pad, ty + pad), label, fill="white", font=font)
    return img   # no matched entities -> returns the plain image (== baseline)


def crop_entities(image_path: Path, detection: dict) -> list[Image.Image]:
    """One crop PER DISTINCT category (non-redundant): keep the highest-scoring exemplar
    of each identity/subcategory, ordered by score. So 5 american + 3 rainbow flags ->
    one (best) american crop + one (best) rainbow crop, not five near-duplicates.
    (Score = face det_score / object GDINO score.) The caller still caps at
    MAX_VISUAL_AUX_CROPS, which now only bites if there are many distinct categories.
    """
    img = Image.open(image_path).convert("RGB")
    best: dict = {}
    for it in matched_entities(detection):
        lab = it["label"]
        if lab not in best or it["score"] > best[lab]["score"]:
            best[lab] = it                                  # highest-scoring exemplar per label
    crops = []
    for it in sorted(best.values(), key=lambda x: x["score"], reverse=True):
        bbox = it.get("bbox")
        if bbox:
            x0, y0, x1, y1 = bbox
            if x1 > x0 and y1 > y0:
                crops.append(img.crop((x0, y0, x1, y1)))
    return crops   # [] if no matched entities -> visual_aux == baseline for that image
