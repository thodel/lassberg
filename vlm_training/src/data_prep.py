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

Usage:
    python vlm_training/src/data_prep.py                  # uses active_preset
    python vlm_training/src/data_prep.py --preset medieval
    python vlm_training/src/data_prep.py --preset all
"""

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

import yaml
from datasets import concatenate_datasets, load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KEEP_COLS    = {"image", "text", "source_type"}
CONFIG_PATH  = Path(__file__).parent.parent / "config" / "datasets.yaml"

# Matches all <Unicode>...</Unicode> blocks in a PageXML string
_UNICODE_RE = re.compile(r"<Unicode>(.*?)</Unicode>", re.DOTALL)


def _extract_pagexml_text(xml: str) -> str:
    """Pull all Unicode text nodes from a PageXML string and join with a space."""
    if not xml:
        return ""
    return " ".join(m.group(1).strip() for m in _UNICODE_RE.finditer(xml) if m.group(1).strip())


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


def _load_one(entry: dict):
    repo_id      = entry["repo_id"]
    source_type  = entry["source_type"]
    min_text_len = entry.get("min_text_len", 3)
    split        = entry.get("split", "train")
    text_column  = entry.get("text_column", "text")

    logger.info(f"Loading {repo_id} ...")
    try:
        ds = load_dataset(repo_id, split=split)
    except Exception as e:
        logger.warning(f"  Skipping {repo_id}: {e}")
        return None

    # ── Normalise text column ─────────────────────────────────────────────────
    if text_column == "xml_content":
        ds = ds.map(
            lambda x: {"text": _extract_pagexml_text(x["xml_content"] or "")},
            desc=f"Extracting text from PageXML ({repo_id})",
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
    val_fraction: float = 0.02,
    seed: int = 42,
    output_dir: str = "data",
):
    entries = load_config(preset)
    parts   = [_load_one(e) for e in entries]
    parts   = [p for p in parts if p is not None]

    if not parts:
        raise RuntimeError("No datasets loaded — check your preset config.")

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
    parser.add_argument("--output_dir",   default="data")
    args = parser.parse_args()

    load_and_prepare(
        preset=args.preset,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
    )
