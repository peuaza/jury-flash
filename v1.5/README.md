# jury-flash v1.5 candidate (2026-11-19)

Long-text poem detector + author-style veto. Not a final release.

## Routing

- `<=14` lines: v1.2 short head / three-state. v1.5 does not score short poems.
- `>=15` lines: v1.5 base (206) + 46-dim author head.

## Clean validation

Real titles used consistently. 4,924 human poems (human1000 + 3,980 modern), 1,810 AI poems
(AI诗/同族/拟目/MiMo/DeepSeek), author 53 held out, external 68 and MiMo-new 35 held out;
46-dim head 5-fold OOF.

| Model | Author all | Author long | External 68 | MiMo-new |
|---|---:|---:|---:|---:|
| base only | 28.3% | - | 82.4% | 100.0% |
| **v1.5 (head 0.8)** | **7.5%** | **4.1% (2/49)** | 79.4% | **97.1%** |

Short author poems: 2/4 flagged, but they go to v1.2, not v1.5.

## Warning

- The earlier 5.7/69.1/94.3 and 5.7/89.7/100 were title-leak artifacts and are retracted
  (`EXPERIMENTS.md` §J.54/§J.57). This v1.5 replaces them.
- External 68 is a short-poem set; use the v1.2 short route for it. The 79.4% figure is the
  long-route diagnostic only, not the short-external product number.
- v1.5 is a candidate; do not tag/publish without the author's review.

## Files

- `_poemguard_v5_206.pkl`
- `_composition_head_v5.npz`
- `score_v15.py`
- `composition_feats.py`
- `METRICS.json`
