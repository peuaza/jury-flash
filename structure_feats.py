# -*- coding: utf-8 -*-
"""§9.156 结构特征统一模块：93 维（五族），**纯文本、不需 GPU**
================================================================================
用法
    import structure_feats as SF
    v = SF.extract(lines)      # -> dict（93 维）
    X = SF.matrix(list_of_lines, keys=SF.KEYS)

五族
    A 信息流      20 维（新信息率的位置剖面、复用结构、首尾呼应）
    B 词汇组织    15 维（hapax/TTR/复现距离/增长速度/zipf）
    C 行内结构    22 维（小句、节奏模板、叙事标记、具体名词代理）
    D 搭配常见度  16 维（相对 poems_human 参考表的二元/三元组频率）
    E 抽象度等    20 维（抽象词、语义场、标点节奏、虚词骨架）
"""
import os, sys
def _base():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


_HERE = _base()
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import infoflow as IF
import ngramfeat as NG

KEYS_A = IF.keys()
KEYS_B = IF.keys_vocaborg()
KEYS_C = IF.keys_linestruct()
KEYS_D = NG.keys()
KEYS_E = IF.keys_semfield()
KEYS = KEYS_A + KEYS_B + KEYS_C + KEYS_D + KEYS_E


def extract(lines, families="ABCDE"):
    """返回 dict（93 维）；任一族失败返回 None"""
    L = [y for y in lines if y and y.strip()]
    out = {}
    if "A" in families:
        v = IF.feats_infoflow(L)
        if v is None: return None
        out.update({k: v[k] for k in KEYS_A})
    if "B" in families:
        v = IF.feats_vocaborg(L)
        if v is None: return None
        out.update({k: v[k] for k in KEYS_B})
    if "C" in families:
        v = IF.feats_linestruct(L)
        if v is None: return None
        out.update({k: v[k] for k in KEYS_C})
    if "D" in families:
        v = NG.feats_ngram(L)
        if v is None: return None
        out.update({k: v[k] for k in KEYS_D})
    if "E" in families:
        v = IF.feats_semfield(L)
        if v is None: return None
        out.update({k: v[k] for k in KEYS_E})
    return out


def matrix(texts, families="ABCDE"):
    import numpy as np
    rows = []
    for L in texts:
        v = extract(L, families)
        if v is None: continue
        rows.append([v[k] for k in KEYS if k in v])
    return np.array(rows)


if __name__ == "__main__":
    import io, json, sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("  五族维度：A%d B%d C%d D%d E%d ⇒ 共 %d 维" % (
        len(KEYS_A), len(KEYS_B), len(KEYS_C), len(KEYS_D), len(KEYS_E), len(KEYS)))
    L = [json.loads(l)["lines"] for l in io.open("jury_samples/human_filtered.jsonl", encoding="utf-8")]
    v = extract(L[4])
    print("  《轻三角》抽取成功：%d 维" % len(v))
