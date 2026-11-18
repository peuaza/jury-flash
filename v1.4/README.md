# jury-flash v1.4 candidate (2026-11-19)

This is a research candidate, not a final release.

## What it is

- `_poemguard_v4_206.pkl`: poem-domain base, 206 features (160 text/structure + 46 poem-organization).
- `_composition_head_v4.npz`: 46-feature author-style head.
- `score_v14.py`: two-stage scorer.

## Clean validation

Real titles are used consistently for training and evaluation. The author's 53 poems are held out;
the 68 external poems and 35 MiMo-new poems are held out from the base; the 46-feature head is
5-fold out-of-fold.

| Model | Author FP | External 68 | MiMo-new |
|---|---:|---:|---:|
| title-consistent base | 34.0% | 85.3% | 100.0% |
| **v1.4 candidate (head threshold 0.7)** | **9.4%** | **80.9%** | **88.6%** |

## Important

- The earlier 5.7/69.1/94.3 and 5.7/89.7/100 results were contaminated by title leakage:
  author filenames were used as titles and training sources had blank titles. They are retracted
  in `EXPERIMENTS.md` §J.54/§J.57 and superseded by §J.59.
- The no-title control is 7.5/63.2/88.6. Title is useful only when it is the real title everywhere.
- This candidate does **not** reach author FP 5%. Do not advertise it as such.

## Routing

For `>=15` lines:

1. base p < 0.298 -> human;
2. author-head p >= 0.70 -> human;
3. otherwise AI.

Short poems still need the v1.2 short head; this candidate is not a replacement for that route.

## Files

- `feature` implementation: `composition_feats.py`
- metrics: `METRICS.json`
- experiment ledger: `../EXPERIMENTS.md` §J.59
