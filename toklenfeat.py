# -*- coding: utf-8 -*-
"""§9.169 token 长度分布族（model-free：**只需分词器，无前向、无 GPU**）
================================================================================
来源：§9.108/§9.109 已验证对「原文 vs AI 逐行改写」有效：
   tl3_frac AUC 0.753 ｜ tl_sd AUC 0.742 ｜ tl_m 0.707 ｜ tok_entropy 0.673
   （参考：旧 57 维在改写任务上仅 0.528）
机制：AI 改写把词「切碎」⇒ token 更短、更均匀
"""
import os, sys, collections, math
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TK = None


class _Tok:
    """轻量封装：只用 `tokenizers`（7.9MB），不依赖 transformers"""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(path)

    def __call__(self, text):
        e = self.t.encode(text or "")
        return {"input_ids": e.ids}

    def decode(self, ids):
        return self.t.decode(list(ids))


def _tk():
    global _TK
    if _TK is None:
        p = os.path.join(_HERE, "tokenizer.json")
        _TK = _Tok(p)
    return _TK

PUNCT = set("，。！？；：、（）《》【】…—-·「」『』“”‘’")
END_P = set("，。！？；：、…—")


def feats_toklen(lines):
    """token 长度分布 + 分片统计（model-free）"""
    L = [x.strip() for x in lines if x and x.strip()]
    if len(L) < 6: return None
    TK = _tk()
    toks = []
    for ln in L:
        try: toks.append(TK(ln)["input_ids"])
        except Exception: toks.append([])
    flat = [i for r in toks for i in r]
    if len(flat) < 20: return None
    out = {}
    # ── token 长度（解码后字符数）
    try:
        dec = np.array([max(1, len(TK.decode([i]).strip())) for i in flat], float)
    except Exception:
        dec = np.ones(len(flat))
    out["tk_m"] = float(dec.mean()); out["tk_sd"] = float(dec.std())
    s = dec.std() + 1e-9
    out["tk_skew"] = float(((dec - dec.mean()) ** 3).mean() / s ** 3)
    out["tk_kurt"] = float(((dec - dec.mean()) ** 4).mean() / s ** 4)
    out["tk1_frac"] = float((dec <= 1).mean())
    out["tk3_frac"] = float((dec >= 3).mean())          # ★ §9.109 最强
    out["tk2_frac"] = float((dec == 2).mean())
    # ── 分片统计
    c = collections.Counter(flat)
    n = len(flat)
    out["tk_ttr"] = len(c) / n
    out["tk_hapax"] = sum(1 for v in c.values() if v == 1) / n
    out["tk_top1"] = max(c.values()) / n
    out["tk_entropy"] = float(-sum((v / n) * math.log(v / n) for v in c.values()))
    # ── 逐行 token 数（长度分布的第二个侧面）
    per = np.array([len(r) for r in toks], float)
    out["tk_per_line_m"] = float(per.mean()) if len(per) else 0.0
    out["tk_per_line_sd"] = float(per.std()) if len(per) else 0.0
    # 每行的 token 长度均值（行间变异）
    lm = np.array([np.mean([max(1, len(TK.decode([i]).strip())) for i in r]) if r else 0.0 for r in toks])
    out["tk_linelen_m"] = float(lm.mean()); out["tk_linelen_sd"] = float(lm.std())
    # ── 行末标点（改写的结构性不变量，作对照）
    ends = np.array([1.0 if (x and x[-1] in END_P) else 0.0 for x in L])
    out["tk_lineend_punct"] = float(ends.mean()); out["tk_lineend_sd"] = float(ends.std())
    pn = np.array([sum(1 for ch in x if ch in PUNCT) for x in L], float)
    out["tk_punct_per_line"] = float(pn.mean()); out["tk_punct_pl_sd"] = float(pn.std())
    return out


def keys():
    L = ["\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b", "\u4e5d\u5341\u7532\u4e59\u4e19\u4e01\u620a\u5df1",
         "\u5e9a\u8f9b\u58ec\u7678\u5b50\u4e11\u5bc5\u536f", "\u8fb0\u5df3\u5348\u672a\u7533\u9149\u620c\u4ea5",
         "\u5b50\u4e11\u5bc5\u536f\u8fb0\u5df3\u5348\u672a", "\u7533\u9149\u620c\u4ea5\u4e00\u4e8c\u4e09\u56db"]
    r = feats_toklen(L)
    return sorted(r.keys()) if r else []


if __name__ == "__main__":
    import io, json
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("  %d 维：%s" % (len(keys()), keys()))
    J = [json.loads(l)["lines"] for l in io.open(os.path.join(_HERE, "jury_samples/human_filtered.jsonl"),
                                                encoding="utf-8")]
    v = feats_toklen(J[4])
    print("  《轻三角》：", {k: round(v[k], 4) for k in list(v)[:8]})
