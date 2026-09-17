# ATR batch — atr_trocr_corpus

- pages: **899**
- started: 2026-09-17T06:30:41+00:00
- wall time: 522.2 min

| model | read | skipped | failed | cut off | chars/page | s/page | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `trocr-kurrent` | 899 | 0 | 0 | 0 | 844 | 38.0 | 522.2 min |

**These columns are not quality.** There is no ground truth in this run, and characters per page measures how much a model wrote, not how much of it is right — a model that hallucinates fluently leads this table. Comparing the readings is the point; the numbers only say what each run cost and whether it completed. Measuring accuracy needs transcribed lines and `eval/linebench.py`.

## Readings

899 page(s) per model — too many to put in one file. The transcriptions are in `/home/dh/agentic_historian/agentic_historian/data/vlm_test/atr_trocr_corpus`, one `.txt` per page per model.
