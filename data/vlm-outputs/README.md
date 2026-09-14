# VLM outputs

Machine transcriptions of Laßberg scans, produced by the
[`agentic_historian`](https://github.com/thodel/agentic_historian) batch runner
against the ATR gateway. **Raw model output** — nothing here has been corrected,
reconciled or reviewed, and none of it is edition text.

## Layout

```
data/vlm-outputs/
└── <run>/                       e.g. atr_test_lassberg
    ├── <model id>/              one directory per model, e.g. qwen3vl-german-xix-v1
    │   ├── <page>.txt           the transcription, nothing else
    │   └── <page>.json          the same text plus provenance (below)
    ├── manifest.jsonl           one line per page attempt, across all models
    ├── report.md                what each model produced and what it cost
    └── report.json              the same, machine-readable
```

`<page>` is the image's path inside the source share with separators folded to
`__`, so `digitalisate/letter-01/001.jpg` becomes `letter-01__001`. The same page
has the same name under every model, which is what makes the directories
comparable line by line — `diff` across two model directories is a diff of two
readings of the same material.

## What a `.json` carries

| field | |
|---|---|
| `text` | the transcription, identical to the `.txt` |
| `model`, `engine` | which model answered, and through which engine |
| `source.sha256`, `source.name`, `source.bytes` | **which bytes were read.** A filename survives re-scanning and re-cropping unchanged; the hash does not, so this is what identifies the image a reading belongs to once the staging directory is gone |
| `timing_ms` | how long the gateway took |
| `lines` | per-line readings with their geometry — for page-level models this is empty, because there was no segmentation step |
| `second_opinion` | party's reading of the same image, attached by the gateway |
| `recognised_at`, `gateway_version`, `schema` | when, by what, in what shape |

## How to read the numbers in `report.md`

Pages read, characters per page, seconds per page. **None of it is quality.**
There is no ground truth in these runs: characters per page says how much a model
wrote, not how much of it is right, and a model that hallucinates fluently leads
that column. Confidence is the model's opinion of itself.

The comparison is the point — several readings of the same page, side by side.
Measuring accuracy is a different job needing transcribed lines; see
`docs/EVALUATION_HARNESS.md` in `agentic_historian`.

## Provenance of the models

The `dh-unibe` German-XIX models were trained on four corpora listed in their
registry entries and on their model cards, among them Transkribus Kurrent-XIX
training sets. Two consequences worth keeping in view:

- the ≈1 % CER on their model cards is each training run's **own** held-out
  split. It says the training converged. It does not transfer to material they
  have not seen, and it is not comparable to any CER measured elsewhere.
- an evaluation set drawn from any of those four corpora measures memorisation,
  not recognition. The Laßberg scans are not among them, which is what makes a
  run over this collection informative.

## Reproducing a run

See [`docs/BATCH_ATR.md`](https://github.com/thodel/agentic_historian/blob/main/docs/BATCH_ATR.md).
Each run directory is self-describing: `report.json` names the models and the
gateway version, and every page records the hash of the image it was read from.
