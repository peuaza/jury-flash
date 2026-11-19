# jury-flash v1.8 candidate (2026-11-19)

Two routes:

- `<=14` lines: `_short_v20.npz` (206 features, no title, no n_lines, no pastiche training).
- `>=15` lines: v1.7 base + 46 author head (`_poemguard_v7_206.pkl`, `_composition_head_v7.npz`).

## Short route (held out)

| Set | n | Detection / FP |
|---|---:|---:|
| external 68 | 68 | **97.1%** |
| pastiche 249 | 249 | **90.0%** |
| short human 571 | 571 | FP **5.1%** |

Per poet: 于坚 82.9, 北岛 100, 多多 93.8, 张枣 91.4, 海子 92.9, 翟永明 82.9,
郑愁予 93.3, 顾城 87.5.

## Long route

Author long FP **0/47**; MiMo-new 100% (head in-sample; OOF 97.1%); external-68 long diagnostic 80.9% (short set; use short route).

## Warnings

- Earlier 100% external / 98% pastiche was a script bug: pastiche was still in `AI.append(P206[tr])`.
  That line is removed; use the numbers above.
- v1.8 is a candidate; not tagged or published.
- Short model includes the author's 4 short poems as human training examples; author-short is therefore in-sample.
