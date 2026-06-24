#!/usr/bin/env bash
# pilot_run.sh — end-to-end pilot test for POLVIFIDEL
#
# Usage:
#   bash pilot_run.sh           # download + run 20 images
#   bash pilot_run.sh 30        # use 30 images
#
# Requires: OPENAI_API_KEY set in environment
#   export OPENAI_API_KEY=sk-...

set -euo pipefail

N=${1:-20}
SCRIPTS="scripts"

echo "======================================================"
echo " POLVIFIDEL Pilot Test  (n=$N images)"
echo "======================================================"

# ── Step 1: Download pilot images ──────────────────────────────────────────
echo ""
echo "[1/5] Downloading $N pilot images from df_2025.csv..."
python "$SCRIPTS/00_download_pilot_sample.py" --n "$N"

# ── Step 2: Entity detection ───────────────────────────────────────────────
echo ""
echo "[2/5] Running entity detection (Grounding-DINO + InsightFace)..."
echo "      (No gallery = face detection only, no identity matching)"
python "$SCRIPTS/02_detect_entities.py"

# ── Step 3: Generate captions (GPT-4o, all 4 conditions) ──────────────────
echo ""
echo "[3/5] Generating captions with GPT-4o (4 conditions × $N images)..."
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-..."
    exit 1
fi
python "$SCRIPTS/04_generate_captions_api.py" --model gpt4o

# ── Step 4: Compute metrics ────────────────────────────────────────────────
echo ""
echo "[4/5] Computing metrics (BLEU, ROUGE, METEOR, CIDEr, VIFIDEL, POLVIFIDEL)..."
python "$SCRIPTS/05_compute_metrics.py" \
    --references data/df_pilot.csv \
    --skip-spice

# ── Step 5: Print summary ──────────────────────────────────────────────────
echo ""
echo "[5/5] Pilot summary (condition comparison):"
echo "------------------------------------------------------"
python "$SCRIPTS/pilot_summary.py"

echo ""
echo "======================================================"
echo " Done. Full metrics at: data/metrics/metrics.csv"
echo "======================================================"
