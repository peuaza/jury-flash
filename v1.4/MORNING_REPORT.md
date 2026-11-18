# Morning report — 2026-11-19

## Bottom line

I did not reach author FP 5% honestly. The cleanest candidate is **9.4% author FP,
80.9% external-68, 88.6% MiMo-new** (head threshold 0.7). Do not release it as a 5% model.

## What changed overnight

- Added MiMo-on/off (169), DeepSeek long (105), AI诗/同族/拟目 to the poem-domain training pool.
- Rebuilt human side from human1000 + modern corpus with real titles (2,934 poems).
- Found and fixed a title leak: earlier 5.7/69.1/94.3 and 5.7/89.7/100 used filenames as
  author titles while training sources had blank titles. Those numbers are retracted in
  EXPERIMENTS.md §J.54/§J.57; clean record is §J.59.

## Clean numbers

| Model | Author FP | External 68 | MiMo-new |
|---|---:|---:|---:|
| title-consistent base (no head) | 34.0% | 85.3% | 100.0% |
| v1.4 candidate, head 0.7 | 9.4% | 80.9% | 88.6% |
| v1.4 candidate, head 0.8 | 9.4% | 82.4% | 97.1% |

The 0.8 head is better on external/MiMo but classifies 《水汽》 as AI; 0.7 keeps it human.
Neither reaches 5%.

## Not to do

- Do not quote 5.7/69.1/94.3 or 5.7/89.7/100; they are title-leak artifacts.
- Do not put doctor/cyto in the paper as detectors.
- Do not advertise v1.4 as final; it is a candidate.

## Next honest options

1. Accept ~9% author FP and publish the tradeoff (no 5% claim).
2. Add more author-style human poems; 53 is still the binding constraint.
3. Add genuinely new poem-level dimensions (semantic organization) and retrain; current semantic
   prototype did not beat the 46-dim organization head.

Files: `_poemguard_v4_206.pkl`, `_composition_head_v4.npz`, `score_v14.py`, `METRICS.json`.
