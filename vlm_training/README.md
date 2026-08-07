# VLM HTR Training Pipeline

QLoRA fine-tuning of [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
for historical handwriting recognition (HTR), using the [dh-unibe](https://huggingface.co/dh-unibe)
datasets as training material.

## Hardware requirements

2x NVIDIA A40 (48 GB each). The model (31B MoE, 3B active) fits via `device_map="auto"` with
4-bit NF4 quantization (~16 GB weights).

## Pipeline overview

```
Incoming JPGs + txts
       |
  ingest_pages.py        preprocess & push to HF Hub
       |
  HF Hub dataset         same schema as dh-unibe line datasets + source_type column
       |
  data_prep.py           merge line + page HF datasets into local Arrow cache
       |
  train.py               QLoRA fine-tune (source-aware collator)
       |
  eval.py                CER / WER on validation split
       |
  push_to_hub.py         upload LoRA adapter to HF Hub
```

## Quick start

All commands are run from the **repo root** (`~/Repo/lassberg`).

```bash
# 1. Create and activate the virtual environment
python3 -m venv vlm_training/.venv
source vlm_training/.venv/bin/activate

# 2. Install dependencies
pip install -r vlm_training/requirements.txt
hf auth login           # needed to pull Qwen3-VL + push results
wandb login             # optional, for training logs

# 3. (Optional) Ingest your own page-level scans and push to HF Hub
python vlm_training/src/ingest_pages.py \
    --input_dir  data/pages_new \
    --repo_id    dh-unibe/image-text_lassberg-letters \
    --collection "Lassberg Letters" \
    --language   de \
    --date_range xix
# Then add the repo_id to the 'custom' preset in vlm_training/config/datasets.yaml

# 4. Prepare training data
# Edit vlm_training/config/datasets.yaml to choose a preset (all / medieval / modern / custom)
# No flags needed on the server — just run:
python vlm_training/src/data_prep.py
#   writes: /mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/data/{train,val}

# 5. Train
python vlm_training/src/train.py

# 6. Evaluate
python vlm_training/src/eval.py \
    --adapter output/qwen3-vl-htr \
    --n_samples 1000

# 7. Push adapter to Hub
python vlm_training/push_to_hub.py \
    --adapter  output/qwen3-vl-htr \
    --repo_id  dh-unibe/qwen3-vl-30b-htr
```

## Storage layout on asterAIx

The NVMe root partition is too small for the HF cache (~660 GB), so it lives on the
network share and is reached through a symlink at the standard cache path:

```bash
~/.cache/huggingface/hub -> /mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub
```

Because it is the *standard* path, no `HF_HOME` is needed — every HF tool (datasets,
transformers, `hf download`) writes there automatically. Verify with:

```bash
ls -la ~/.cache/huggingface/hub && df -h /
```

`TMPDIR` must point at the share too, and it has to be exported **before** Python starts
(`dill` reads it at import time). Add this to `~/.bashrc`:

```bash
export TMPDIR=/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/tmp
```

## Choosing datasets

Datasets are configured in `vlm_training/config/datasets.yaml`. Set `active_preset` to one of
the built-in presets, or pass `--preset` on the command line:

```bash
python vlm_training/src/data_prep.py --preset medieval
python vlm_training/src/data_prep.py --preset modern
python vlm_training/src/data_prep.py --preset all      # default
python vlm_training/src/data_prep.py --preset custom   # your own page-level data
```

To add a new dataset, append it to any preset in `datasets.yaml`:

```yaml
- repo_id: dh-unibe/image-text_my-collection
  source_type: page       # or "line" for text-line crops
  text_column: text       # or "xml_content" for PageXML datasets
  min_text_len: 20
```

## Dataset sources

### Line-level (from dh-unibe HF Hub)

| Dataset | Samples | Content |
|---|---|---|
| image-text_kurrent-xix | ~158K | 19th-c. Kurrent handwriting |
| image-text_medieval-scripts_xiv-xv-xvi | ~100K+ | Medieval Latin/German |
| image-text_zh-regierungsratsprotokolle | ~100K+ | Zurich council minutes |
| image-text_historisches-grundbuch-basel_xix-xx_train | ~10K | Basel property register (ground truth subset) |
| image-text_aaeb-xiv-xvii | ~2.5K | Cantonal archive, multilingual |
| image-text_aaeb-xiv-xvii-part-2 | ~121 | Cantonal archive part 2 (PageXML) |
| image-text_parzival-part-1 | ~3.6K | Medieval manuscript |
| image-text_rats-und-richtebuecher_xv-xvi | ~10K+ | Council records |
| image-text_german-20th-century | ~8.5K | 20th-c. German handwriting |
| image-text_koenigsfelden-charters-part-2 | ~68 | Königsfelden charters |
| image-text_koenigsfelden-charters-part-3 | ~n<1K | Königsfelden charters |
| image-text_koenigsfelden-charters-post-1500 | ~1K+ | Königsfelden post-1500 |
| image-text_koenigsfelden-adhr-colmar | ~223 | Middle High German/Latin charters (PageXML) |
| image-text_hgb-kf_mixture | ~154 | Mixed Basel/Königsfelden (PageXML) |
| image-text_nr-sr-vereinigte-bundesversammlung-xix | ~182 | Swiss parliament minutes (PageXML) |
| data-towerbooks-textlines | ~47.8K | Tower books text lines |

Excluded from training:
- `image-text_handwritten-bundesratsprotokolle_xix-xx` — auto-transcribed, no ground truth
- `image-text_historisches-grundbuch-basel_xix-xx` (full) — auto-transcribed, no ground truth

### Page-level (uploaded via ingest_pages.py)

Add your page-level datasets to the `custom` preset in `vlm_training/config/datasets.yaml`.

## Source-aware collator

The `HTRCollator` applies different visual token budgets per sample type:

| source_type | max_pixels | max_seq_len | Use case |
|---|---|---|---|
| `line` | 256 × 28² ≈ 200K | 512 | Text-line crops |
| `page` | 2048 × 28² ≈ 1.6M | 4096 | Full-page scans |

## Memory budget (2x A40)

| Component | QLoRA (default) |
|---|---|
| Model weights (4-bit NF4) | ~16 GB |
| LoRA adapters | ~0.3 GB |
| Activations + grad checkpointing | ~8 GB |
| Optimizer (paged 8-bit Adam) | ~2 GB |
| **Total** | **~26 GB (single GPU)** |
