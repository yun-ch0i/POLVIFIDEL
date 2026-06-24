"""
Central configuration: paths and experiment settings.
Paths adapt automatically between local and cluster environments.
"""
import os
from pathlib import Path

# Detect environment: set POLVIFIDEL_ROOT on the cluster via SLURM script,
# otherwise fall back to the repo root relative to this file.
_REPO_ROOT = Path(os.environ.get("POLVIFIDEL_ROOT", Path(__file__).parent.parent))

# --- Directories ---
DATA_DIR    = _REPO_ROOT / "data"
IMAGES_DIR  = DATA_DIR / "images"          # raw downloaded images
DETECTIONS_DIR = DATA_DIR / "detections"   # per-image detection outputs (JSON)
CAPTIONS_DIR   = DATA_DIR / "captions"     # per-model caption outputs (CSV)
METRICS_DIR    = DATA_DIR / "metrics"      # computed metric scores (CSV)

GALLERY_DIR = DATA_DIR / "gallery"
OBJECT_CLUSTERS_DIR = DATA_DIR / "object_clusters"   # build_object_gallery.ipynb outputs

for _d in [IMAGES_DIR, DETECTIONS_DIR, CAPTIONS_DIR, METRICS_DIR, GALLERY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# --- Princeton AI Sandbox (via the Portkey AI Gateway) ---
# All `type: "api"` models below are served by AI Sandbox through Portkey — no GPU,
# no per-call billing. Auth is a single key in the AI_SANDBOX_KEY env var; the
# gateway handles routing, so no base_url / api_version is needed.
#   export AI_SANDBOX_KEY=<key provided to you>
AI_SANDBOX_KEY = os.environ.get("AI_SANDBOX_KEY", "")

# --- Experiment design ---
# type "api"   -> served by AI Sandbox, routed through one OpenAI-compatible client
# type "local" -> self-hosted on a GPU via SLURM (pinned HuggingFace checkpoints)
# model_id for api models must match the exact Sandbox roster name. Image-capable
# Sandbox models: gpt-5-mini, gpt-4o-mini, gemini-3.1-pro-preview, claude-opus-4-7,
# claude-sonnet-4-6, claude-haiku-4-5, Llama-4-Scout-17B-16E-Instruct,
# mistral-small-2503, mistral-medium-2505.
MODELS = {
    # --- Proprietary, via Sandbox (PAID — counts against the $250/mo cap) ---
    # Budget decision: benchmark only GPT + Gemini. Roster has no full gpt-4o or
    # gemini-2.5, so these are remapped to current IDs. Uncomment to re-enable.
    "gpt4o":         {"type": "api",   "model_id": "gpt-4o-mini"},          # or "gpt-5-mini"
    "gemini":        {"type": "api",   "model_id": "gemini-3.1-pro-preview"},
    # "claude-opus":   {"type": "api",   "model_id": "claude-opus-4-7"},
    # "claude-sonnet": {"type": "api",   "model_id": "claude-sonnet-4-6"},
    # "claude-haiku":  {"type": "api",   "model_id": "claude-haiku-4-5"},
    # "llama4-scout":  {"type": "api",   "model_id": "Llama-4-Scout-17B-16E-Instruct"},
    # "mistral-small": {"type": "api",   "model_id": "mistral-small-2503"},
    # --- Open-source, self-hosted on cluster GPU (pinned checkpoints; no API cost) ---
    # LLaVA-1.6 removed: single-image model -> unreliable multi-image visual_aux.
    "internvl":      {"type": "local", "model_id": "OpenGVLab/InternVL2-8B"},
    "qwen":          {"type": "local", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
}

# Convenience groupings for the two run routes.
API_MODELS   = [m for m, c in MODELS.items() if c["type"] == "api"]
LOCAL_MODELS = [m for m, c in MODELS.items() if c["type"] == "local"]

CONDITIONS = ["baseline", "visual_aux", "textual_aux", "annotated_aux"]

# --- Face recognition ---
FACE_QUALITY_THRESHOLD = 0.5     # InsightFace det_score; below this → skip matching
FACE_SIMILARITY_THRESHOLD = 0.4  # cosine similarity; below this → "unknown"

# --- Grounding-DINO ---
GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"  # swap to 'base' for higher recall
GDINO_BOX_THRESHOLD = 0.35
GDINO_TEXT_THRESHOLD = 0.25
POLITICAL_OBJECT_QUERIES = [
    "flag", "banner", "sign", "poster", "podium", "microphone",
    "hat", "badge", "button", "campaign shirt", "protest sign",
]

# --- Political-object gallery (CLIP subcategory matching in 02_detect_entities) ---
# OBJECT_GALLERY_CLIP MUST match the checkpoint used to BUILD the gallery
# (scripts/build_object_gallery.ipynb), or the cosine space won't align.
OBJECT_GALLERY_CLIP = "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"
OBJECT_SIMILARITY_THRESHOLD = 0.75   # cosine; CLIP image sims run high — CALIBRATE on a sample

# --- Sampling ---
N_PER_CELL = 150   # target images per experimental cell
BETA = 1.0         # POLVIFIDEL hallucination penalty exponent (tune later)
