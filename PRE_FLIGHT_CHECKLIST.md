# Pre-Flight Checklist — Caption Generation

Run **every** check below on a tiny sample before launching caption generation on
the full corpus (1,200 images × 4 conditions × 10 models). Each failed full run
wastes Sandbox quota, GPU node-hours, and (worse) produces captions you can't trust
in the metrics. The whole checklist should take well under an hour and saves days.

Two routes produce captions into the **same** `data/captions/{model}_{condition}.csv`:

| Route | Script | Models | Where it runs |
|-------|--------|--------|----------------|
| API | `04_generate_captions_api.py` | gpt4o, gemini, claude-opus/sonnet/haiku, llama4-scout, mistral-small | login node / laptop (no GPU) |
| Cluster | `04_generate_captions_cluster.py` | llava, internvl, qwen | GPU via `cluster/run_vlm_inference.slurm` |

> Prompts and the four conditions live in `caption_common.py` and are imported by
> both scripts, so they are **guaranteed identical** across all models. Do not
> redefine prompts in either script.

---

## Phase 0 — Environment & credentials

### API route (AI Sandbox)
- [ ] Sandbox account approved; `SANDBOX_BASE_URL`, `SANDBOX_API_KEY`, and (if Azure-style) `SANDBOX_API_VERSION` exported.
- [ ] Confirm the **exact** Sandbox model strings for `gpt4o` and `gemini` against the live roster and fix `config.py` if they differ (the Claude/Llama/Mistral IDs were copied verbatim from the roster).
- [ ] Client connects: a one-line text-only chat completion to **each** of the 7 models returns 200 (catches auth, wrong base_url/version, and any model name Sandbox doesn't actually expose).
- [ ] `AzureOpenAI` vs `OpenAI` path: verify `get_sandbox_client()` picked the right one for how Sandbox is fronted (depends on whether `SANDBOX_API_VERSION` is set).

### Cluster route (GPU)
- [ ] `conda activate polvifidel` env exists on the cluster with `transformers`, `torch`, `torchvision`, `accelerate`, `timm`, `einops`, `sentencepiece`.
- [ ] `transformers` is new enough for **Qwen2.5-VL** (`Qwen2_5_VLForConditionalGeneration` importable) and **LLaVA-NeXT**.
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True` on a GPU node (run inside an interactive `salloc`, not the login node).
- [ ] InternVL2 loads with `trust_remote_code=True` (it downloads custom modeling code — confirm the cluster allows this / is online or pre-cached).
- [ ] Weights are downloadable or pre-staged in the HF cache (cluster compute nodes are often offline — pre-download on the login node and set `HF_HOME`).

---

## Phase 1 — Data integrity

- [ ] `data/images/` populated; count matches the manifest you intend to caption.
- [ ] `data/detections/{image_id}.json` exists for those images (captions still run without detections, but `textual/visual/annotated_aux` degrade to "no entities").
- [ ] **Detection JSON schema matches what the caption code expects:**
  - `faces`: list of objects each with a `name` and a `bbox`.
  - `objects`: list of objects each with a `label` and a `bbox`.
  - `bbox` is `[x0, y0, x1, y1]` in **absolute pixel** coordinates (this is what `crop_entities` and `draw_annotations` assume — if detection wrote normalized or `[x,y,w,h]` boxes, crops/annotations will be garbage).
- [ ] `image_id` in detection filenames == image file stem == what ends up in the CSV `image_id` column (so metrics can join them).

---

## Phase 2 — Per-model smoke test (THE important one)

Run **2–5 images, all four conditions**, for **every** model. Read the captions by eye.

```bash
# API — repeat per model
python scripts/04_generate_captions_api.py --model gpt4o        --limit 3
python scripts/04_generate_captions_api.py --model claude-sonnet --limit 3
python scripts/04_generate_captions_api.py --model llama4-scout  --limit 3
# ... all 7

# Cluster — repeat per model (on a GPU node)
python scripts/04_generate_captions_cluster.py --model qwen     --limit 3
python scripts/04_generate_captions_cluster.py --model llava    --limit 3
python scripts/04_generate_captions_cluster.py --model internvl --limit 3
```

For each model, confirm:
- [ ] All four conditions produce a **non-empty, coherent** caption (not an error string, not an empty cell, not a refusal).
- [ ] The caption describes the **actual image** (not a hallucinated/echoed prompt).
- [ ] **`visual_aux` multi-image actually works** — this is the highest-risk check:
  - API: confirm Sandbox accepts multiple `image_url` parts for **Claude** and **Llama-4-Scout** specifically (some endpoints cap or reject multi-image). If rejected, decide: skip visual_aux for that model, or send crops differently.
  - Cluster: **LLaVA-1.6 is single-image** — inspect whether multi-image visual_aux output is sensible or degenerate; document the decision either way. Qwen/InternVL handle multi-image natively but still eyeball them.
- [ ] Captions respect the length cap (~100 words / 3 sentences) and aren't being cut off mid-word by `MAX_CAPTION_TOKENS=160` (raise it if they consistently truncate).
- [ ] GPU memory: each cluster model fits in the requested `--mem`/VRAM without OOM. Note peak usage to size the SLURM job.

---

## Phase 3 — Output & resume consistency

- [ ] CSV columns are identical across both routes: `image_id, model, condition, caption, prompt_tokens, completion_tokens, timestamp`.
- [ ] `prompt_tokens`/`completion_tokens` populated where available (InternVL leaves them null by design — fine).
- [ ] **Checkpoint/resume works:** re-run the same smoke-test command and confirm it reports "N done, 0 remaining" and does **not** duplicate rows. (Both routes skip already-captioned `image_id`s.)
- [ ] No stray partial rows from an interrupted run (append-mode means a kill mid-write can leave one bad line — spot check the tail of a CSV).

---

## Phase 4 — Downstream metric dry-run

- [ ] Run `05_compute_metrics.py` on the smoke-test captions end-to-end (catches schema/join bugs now, not after the full run).
- [ ] Entity-dictionary alignment produces non-trivial matches on the sample (the dictionary route, not CLIP).
- [ ] **Floor-effect sanity check** (known risk for this project): POLVIFIDEL/VIFIDEL scores are not all identical or all ~0 across the sample. If every caption scores the same, the metric or the alignment dict — not the models — is the problem; fix before scaling.

---

## Phase 5 — Scale & cost projection

- [ ] **API:** from the smoke test, estimate total calls = images × conditions × 7 models. Check Sandbox rate limits / any daily quota; tune `--workers` so you don't trip 429s. (Free, but quota/rate still bite.)
- [ ] **Cluster:** measure seconds/image/model from the smoke test → set `--time` in `run_vlm_inference.slurm` with headroom (the array runs llava/internvl/qwen in parallel, so size for the slowest single model, not the sum).
- [ ] Disk: estimate caption CSV + any cached images/weights footprint against your quota.
- [ ] Decide the **run order**: cheapest/fastest model first as a full-corpus canary before committing all 10.

---

## Go / No-Go

Launch the full corpus only when, for **every** model:
1. All four conditions return coherent captions on the smoke sample, **and**
2. visual_aux multi-image behavior is verified or explicitly documented as a limitation, **and**
3. `05_compute_metrics.py` runs clean on the sample with no floor effect, **and**
4. resume/checkpoint confirmed working.

Then:
```bash
# API (login node) — all 7 models, full corpus
python scripts/04_generate_captions_api.py --model all

# Cluster — array job, llava/internvl/qwen in parallel
sbatch scripts/cluster/run_vlm_inference.slurm
```
