# jury-flash Release Manifest

**Date**: 2026-11-18
**Status**: release-candidate manifest; v1.2.0-rc adds a three-state short-text router; paper instrument remains v1.0.0/v1.0.1.
**Purpose**: map every release artifact and every headline table to a file, a feature-set size, a decision rule, a threshold and a hash.

---

## 1. Artifacts

| Artifact | Size | SHA-256 (first 16) | Role |
|---|---|---|---|
| `_det_multi14.pkl` | 531 KB | `4cf1eceb4f6166cb` | v1.0.0 / v21 main model (paper instrument) |
| `_det_multi14_v22d.pkl` | 531 KB | `100c8189973e81af` | v1.1.0-rc repair candidate |
| `_det_multi14_v23.pkl` | 531 KB | `b9581ccf9658bf83` | v1.2.0-rc general model (≥15 lines) |
| `_short_head_14_np.pkl` | 6 KB | `38e7da1947e39451` | v1.2.0-rc short head (6–14 lines) |
| `_det_multi14_nopriv.pkl` | 531 KB | `2cf1de3211edb35d` | no-private variant (excludes all 53 author poems) |
| `_det_struct.pkl` | 1004 KB | `c7fe5d70f172ef96` | 94-dim specialist head (MLP×5) |
| `feature_dict.csv` | 160 rows | — | key, family, definition |
| `jury-flash-v1.2.0-rc.zip` | 22.3 MB | — | v1.2.0-rc full runnable runtime (code + tokenizer + LM + models) |
| `jury_conf_v12.json` | 1 KB | — | v1.2.0-rc route, thresholds and FP table |
| `score_v12.py` | 5 KB | — | v1.2.0-rc three-state CLI |

Hashes are SHA-256 truncated to 16 hex characters for readability; full hashes can be regenerated with
`sha256sum`. **Aliases** (byte-identical, verified): `_det_multi14_v23.pkl` = `_def_mlp_release_v23.pkl`;
`_det_multi14_nopriv.pkl` = `_def_mlp_public_v22d_vol1mix.pkl`.

## 2. Version table

| Release | Head | Training samples | Decision | Thresholds | Notes |
|---|---|---|---|---|---|
| v1.0.0 (v21) | 160-dim, 14-class MLP×3 | ≈23.8k | argmax ∈ AI **or** AI-class total ≥ `pthr_total` | 0.70 | paper instrument; exam 16/18 default, 17/18 at length-matched FP5 |
| v1.0.1 | code only | — | routing wrapper for local `jury_model.pkl` (`route_min_lines=25`) | — | no weight change |
| v1.1.0-rc | 160-dim, 14-class MLP×3 | 25,833 | same as v1.0.0, plus per-length thresholds | 6–7: 1.000; 8–9: 0.977; 10–14: 0.502; 15+: 0.117 | v22d repair candidate |
| v1.2.0-rc | 160-dim general + binary short head | 25,684 + 2,139 human / 1,451 machine (short head) | ≤14: short head and general agree, else uncertain; ≥15: general confidence 3-state | short head: 6–9 0.686; 10–14 0.556; general: 0.70 / 0.30 | external 45 AI / 7 human / 16 uncertain; raw short head 84.1% (53/63, decontaminated) |
| no-private | 160-dim, 14-class MLP×3 | ≈25.6k | same as v1.0.0 | 0.70 | excludes all 53 author poems; 29 published + augmentation |
| specialist | 94-dim, MLP×5 | see §J.11 | 94-dim score ≥ threshold | FP1 0.6241 / FP3 0.2380 / FP5 0.1188 / FP10 0.0303 | exam 15/18 at FP5 |

**Training-row counts refer to different objects**: 4,386 is the 14-class classification set (§J.12);
≈23.8k is the v21 160-dim training matrix; 25,833 and 25,684 are the v22d and v23 matrices; the public
`_train_bank.npz` is a 14,938×145 auxiliary bank, not the main training set.

## 3. Decision rules

**160-dim general model**

```text
p_ai_total = sum of the seven AI-class probabilities
AI  iff  argmax is an AI class  OR  p_ai_total >= pthr_total (0.70)
```

`pthr_c` is absent from the released 160-dim weights (default 0.0); the `auto_c` variant used in earlier
experiments is not the released rule.

**Short head (v1.2.0-rc)**

```text
6–9 lines:   AI iff sigmoid(w·z + b) >= 0.686
10–14 lines: AI iff sigmoid(w·z + b) >= 0.556
z = (x - mean) / scale; 160 features in feature_dict.csv order
```

**v1.2 default (three-state)**

```text
<=14 lines:  AI      if short head and general both say AI
             human   if both say human
             uncertain otherwise
>=15 lines:  AI if p_ai_total >= 0.70; human if p_ai_total < 0.30; uncertain otherwise
```

## 4. Feature set

160 dimensions: 21 information-flow + 15 vocabulary organization + 22 line structure + 16 n-gram
collocation + 20 semantic-field/function-word/punctuation = **94 structural**; plus 10 character-level
`cz_*` + 19 token-length `tk*` + 22 token-level LM `tl_*` = **51 token/character**; plus 15 sequential-LM
`sl_*` = **160**. The 94-dim specialist head uses the first five families. `feature_dict.csv` lists every
key and family; 37 keys have one-line definitions and the rest point to the implementation in
`infoflow.py`, `ngramfeat.py`, or `cpu_proxy2.py`.

## 5. Corpora and licences

| Source | Use | Licence | Redistribution |
|---|---|---|---|
| `sheepzh/poetry` | 81,327 modern Chinese poems (human) | MIT | yes, with attribution |
| `chinese-poetry` | classical Chinese | public domain | yes |
| 人民网 (scraped) | 223 news documents | copyrighted | **no**; source URL + date only |
| 52poke wiki | 207 expository documents | **CC BY-NC-SA 3.0** | scripts only (non-commercial, share-alike) |
| LCCC | 1,600 chat documents | **MIT** (thu-coai/lccc) | yes with attribution; kept out of the public bundle for privacy |
| CNKI abstracts | 1,050 academic abstracts | copyrighted | **no** |
| letters / books | 1,600 letter texts | copyrighted | **no** |
| official documents (`salpt/chinatax`) | 6,450 paragraphs | **no licence declared** (government public information; verify) | scripts only |
| `mxlimbo/book-cn` | translations/history/religion | **no licence declared** (GitHub API returns null) | **no** |
| author's poems | 53 (29 published, 24 unpublished) | author's own | **no text**; weights may include them (data card) |
| AI-generated corpus | ~4,300 texts; 2,243 in seven genre pools; 473 same-family; exam 18 | generated by the project | feature vectors + labels; texts per §J.10 policy |

**NOTICE reconciliation** (2026-11-17): 52poke is **CC BY-NC-SA 3.0**, not plain CC BY-SA; Wikipedia is a
separate source (CC BY-SA 4.0) and is now listed separately in `NOTICE`. LCCC is **MIT** per
`thu-coai/lccc`; it is used for research and kept out of the public bundle for privacy. `book-cn` has no
declared licence and is not redistributed.

**Weights licence.** Code is MIT. Model weights are released for **research / non-commercial use** because
the training corpus includes 52poke (CC BY-NC-SA 3.0) and copyrighted CNKI/letter text; the MIT code
licence does not cover the weights.

## 6. Reproduction commands

```bash
# exam table
python _score_exam.py _det_multi14.pkl        # 160-dim models
python _score_struct_exam.py                    # 94-dim specialist head

# DET + Wilson/bootstrap CI (needs the feature caches)
python _det_ci.py

# v1.2 router
python score_v12.py poem.txt
```

The feature caches (`_feat_all.pkl`, `_pastiche_feats.npz`, `_shorthuman_feats.npz`, `_human_bands_feats.npz`)
are currently local and **not** part of the release; packaging them (or a feature-bank export without texts)
is the remaining reproducibility task.

## 7. Repository and DOI

- Repository: `https://github.com/peuaza/jury-flash` (origin configured; **no tag pushed yet**).
- DOI: not assigned; plan is a Zenodo archive of the tagged release.
- Model files are not committed to git (see `MODELS.md`); they are distributed via release archives.

## 8. Known gaps

1. Feature bank not yet packaged; `feature_dict.csv` is a key list, not a full formula spec.
2. No tag/DOI yet.
3. Licence status: 52poke CC BY-NC-SA 3.0; LCCC MIT; CNKI/letters copyrighted; chinatax and book-cn undeclared (not redistributed).
4. No short head for ≥15 lines.
5. External evaluation n = 68; per-band CIs are wide.
6. The paper's reported instrument is v1.0.0/v1.0.1; v1.1/v1.2 are candidates and must be labelled as
   such until the paper is finalized.
