"""
Load and combine HF datasets into a unified local Arrow cache.

Datasets are configured in vlm_training/config/datasets.yaml.
Select a preset by setting active_preset in that file, or pass
--preset on the command line to override.

Output columns: image, text, source_type
source_type in {'line', 'page'}

Supported dataset schemas
─────────────────────────
text_column: "text"         Plain text column (most datasets)
text_column: "xml_content"  PageXML — text is extracted from
                            <TextEquiv><Unicode>...</Unicode></TextEquiv>

Storage layout
──────────────
The HF download cache uses the standard location, ~/.cache/huggingface.
On asterAIx that path is a symlink to the network share:

  ~/.cache/huggingface/hub -> /mnt/wbkolleg_dh_1/Textrecognition_Training/hf_hub/
    datasets--dh-unibe--image-text_kurrent-xix/
    datasets--dh-unibe--image-text_medieval-scripts_xiv-xv-xvi/
    models--Qwen--Qwen3-VL-30B-A3B-Instruct/
    ...

Pass --hf_cache only if you need a non-standard cache root.

Arrow output and scratch go to the network share directly:

  /mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder/
    data/
      train/                   ← Arrow cache written by this script
      val/
    tmp/                       ← TMPDIR

Usage:
    python vlm_training/src/data_prep.py                  # uses defaults + active_preset
    python vlm_training/src/data_prep.py --preset medieval
    python vlm_training/src/data_prep.py --preset all
"""

import argparse
import logging
import os
import re
from collections import Counter
from pathlib import Path

import yaml
from datasets import concatenate_datasets, load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KEEP_COLS   = {"image", "text", "source_type"}
CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.yaml"

BASE_DIR    = Path("/mnt/wbkolleg_dh_1/Textrecognition_Training/training_folder")
DEFAULT_OUTPUT  = str(BASE_DIR / "data")

# Standard HF cache root. On asterAIx ~/.cache/huggingface/hub is a symlink
# to the network share, so nothing needs to be overridden here.
DEFAULT_HF_HOME = Path(
    os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
)

# Matches all <Unicode>...</Unicode> blocks in a PageXML string
_UNICODE_RE = re.compile(r"<Unicode>(.*?)</Unicode>", re.DOTALL)


def _repo_cache_dir(hf_home: Path, repo_id: str) -> Path:
    """Return the expected HF hub cache folder for a dataset repo."""
    safe = repo_id.replace("/", "--")
    return hf_home / "hub" / f"datasets--{safe}"


def _extract_pagexml_text(xml: str) -> str:
    """Pull all Unicode text nodes from a PageXML string and join with a space."""
    if not xml:
        return ""
    return " ".join(
        m.group(1).strip() for m in _UNICODE_RE.finditer(xml) if m.group(1).strip()
    )


def load_config(preset: str | None = None) -> list[dict]:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    preset = preset or config["active_preset"]
    entries = config["presets"].get(preset)
    if entries is None:
        available = list(config["presets"].keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    logger.info(f"Using preset: '{preset}'  ({len(entries)} datasets)")
    return entries


def _load_one(entry: dict, hf_home: Path):
    repo_id      = entry["repo_id"]
    source_type  = entry["source_type"]
    min_text_len = entry.get("min_text_len", 3)
    split        = entry.get("split", "train")
    text_column  = entry.get("text_column", "text")

    # ── Skip if already cached locally (same name = same dataset) ────────────
    cache_dir = _repo_cache_dir(hf_home, repo_id)
    if cache_dir.exists():
        logger.info(f"Using cached  {repo_id}  ({cache_dir})")
    else:
        logger.info(f"Downloading   {repo_id} ...")

    try:
        ds = load_dataset(repo_id, split=split)
    except Exception as e:
        logger.warning(f"  Skipping {repo_id}: {e}")
        return None

    # ── Normalise text column ─────────────────────────────────────────────────
    if text_column == "xml_content":
        ds = ds.map(
            lambda x: {"text": _extract_pagexml_text(x["xml_content"] or "")},
            desc=f"Extracting PageXML text ({repo_id})",
        )
    elif text_column != "text":
        ds = ds.rename_column(text_column, "text")

    # ── Filter short / empty texts ────────────────────────────────────────────
    before = len(ds)
    ds = ds.filter(
        lambda x: bool(x["text"]) and len(x["text"].strip()) >= min_text_len
    )
    logger.info(f"  {len(ds):,} rows  (dropped {before - len(ds):,} short/empty texts)")

    # ── Inject source_type if absent ──────────────────────────────────────────
    if "source_type" not in ds.column_names:
        ds = ds.map(lambda _: {"source_type": source_type}, batched=False)

    # ── Drop everything except the three columns we need ─────────────────────
    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    if drop:
        ds = ds.remove_columns(drop)

    return ds


def load_and_prepare(
    preset: str | None = None,
    hf_home: Path = DEFAULT_HF_HOME,
    val_fraction: float = 0.02,
    seed: int = 42,
    output_dir: str = DEFAULT_OUTPUT,
):
    entries  = load_config(preset)
    results  = [_load_one(e, hf_home) for e in entries]
    parts    = [p for p in results if p is not None]
    n_failed = len(results) - len(parts)

    if not parts:
        raise RuntimeError(
            f"No datasets loaded — all {n_failed} dataset(s) failed. "
            "Common causes: no disk space or network errors. "
            f"Check that {hf_home / 'hub'} has sufficient free space."
        )

    if n_failed:
        logger.warning(f"{n_failed} dataset(s) skipped — continuing with {len(parts)} loaded.")

    full = concatenate_datasets(parts).shuffle(seed=seed)
    logger.info(f"Total samples: {len(full):,}")
    for k, v in Counter(full["source_type"]).items():
        logger.info(f"  {k:6s}: {v:,}")

    splits   = full.train_test_split(test_size=val_fraction, seed=seed)
    train_ds = splits["train"]
    val_ds   = splits["test"]

    train_ds.save_to_disk(f"{output_dir}/train")
    val_ds.save_to_disk(f"{output_dir}/val")
    logger.info(
        f"Saved -> {output_dir}/train ({len(train_ds):,}) "
        f"and {output_dir}/val ({len(val_ds):,})"
    )
    return train_ds, val_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        default=None,
        help="Override active_preset from datasets.yaml (e.g. medieval, modern, all, custom)",
    )
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
        help=f"Where to write the Arrow train/val cache (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--hf_cache",
        default=None,
        help="Override the HF cache root (sets HF_HOME). Normally unnecessary — "
             f"defaults to {DEFAULT_HF_HOME}, whose hub/ is symlinked to the network share.",
    )
    args = parser.parse_args()

    # Only override the HF cache when explicitly asked; the default location is
    # already redirected to the network share via the ~/.cache/huggingface/hub symlink.
    hf_home = Path(args.hf_cache) if args.hf_cache else DEFAULT_HF_HOME
    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    os.environ.setdefault("TMPDIR", str(BASE_DIR / "tmp"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    logger.info(f"HF cache      -> {hf_home / 'hub'}")
    logger.info(f"TMPDIR        -> {os.environ['TMPDIR']}")
    logger.info(f"output_dir    -> {args.output_dir}")

    load_and_prepare(
        preset=args.preset,
        hf_home=hf_home,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
    )
