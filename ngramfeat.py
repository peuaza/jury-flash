# -*- coding: utf-8 -*-
"""§9.154 「搭配常见度」特征族（需要 _ngram_ref.pkl 参考表）
================================================================================
角度：把一首诗的字符二元组/三元组拿出来，问「这些搭配在**正常中文诗**里常见吗」
   · 参考表：从 poems_human 的 20000 首人类诗统计（50.9 万二元组）
特征：
   A 搭配常见度：二元组对数频率的均值/离散/分位、未见比例、罕见比例、常见比例
   B 篇内字组重复：同一二元/三元组在篇内重复的比例
   C 虚词搭配多样性：虚词后接字的分布熵
⚠ 风险：这是「参考语料统计」类特征，可能混入**来源差异**（§9.107）
   ⇒ 必须做同来源检验
"""
import io, os, pickle, sys, collections
import numpy as np

def _base():
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


_HERE = _base()
_REF = None
def _ref():
    global _REF
    if _REF is None:
        _REF = pickle.load(io.open(os.path.join(_HERE, "_ngram_ref.pkl"), "rb"))
    return _REF

CJK = lambda ch: "\u4e00" <= ch <= "\u9fff"
FUNC = set("的了在是我你他她它们这那有和与就不人都一也又被把从对到而之其所以为着过很"
           "会能可要还只让给跟向于并且但或如果因为呢吗吧啊呀哦嗯着了个地得")


def feats_ngram(lines):
    L = [x.strip() for x in lines if x and x.strip()]
    if len(L) < 6: return None
    cs = [c for c in "".join(L) if CJK(c)]
    if len(cs) < 20: return None
    R = _ref(); BG = R["bg"]; TG = R["tg"]; TOT = max(1, R["total"])
    out = {}
    # ── A 搭配常见度
    bgs = [cs[i] + cs[i + 1] for i in range(len(cs) - 1)]
    lf = np.array([np.log((BG.get(b, 0) + 0.5) / TOT) for b in bgs])
    out["ng_bg_lf_m"] = float(lf.mean()); out["ng_bg_lf_sd"] = float(lf.std())
    out["ng_bg_lf_p10"] = float(np.percentile(lf, 10)); out["ng_bg_lf_p90"] = float(np.percentile(lf, 90))
    out["ng_bg_lf_min"] = float(lf.min()); out["ng_bg_lf_max"] = float(lf.max())
    unseen = np.array([1.0 if b not in BG else 0.0 for b in bgs])
    out["ng_bg_unseen"] = float(unseen.mean())
    cnt = np.array([BG.get(b, 0) for b in bgs])
    out["ng_bg_rare"] = float((cnt < 10).mean())          # 出现 <10 次的搭配
    out["ng_bg_common"] = float((cnt >= 1000).mean())     # 很常见的搭配
    # 三元组未见比例
    tgs = [cs[i] + cs[i + 1] + cs[i + 2] for i in range(len(cs) - 2)]
    out["ng_tg_unseen"] = float(np.mean([1.0 if t not in TG else 0.0 for t in tgs])) if tgs else 0.0
    # ── B 篇内重复
    cb = collections.Counter(bgs)
    out["ng_bg_rep"] = float(np.mean([1.0 if v > 1 else 0.0 for v in cb.values()])) if cb else 0.0
    ct = collections.Counter(tgs)
    out["ng_tg_rep"] = float(np.mean([1.0 if v > 1 else 0.0 for v in ct.values()])) if ct else 0.0
    # 二元组频率分布的熵（篇内）
    if cb:
        p = np.array(list(cb.values()), float); p /= p.sum()
        out["ng_bg_ent"] = float(-(p * np.log(p + 1e-12)).sum())
    else:
        out["ng_bg_ent"] = 0.0
    # ── C 虚词搭配多样性：虚词后接字的分布熵
    nxt = collections.Counter()
    nf = 0
    for i in range(len(cs) - 1):
        if cs[i] in FUNC:
            nxt[cs[i + 1]] += 1; nf += 1
    if nf:
        p = np.array(list(nxt.values()), float); p /= p.sum()
        out["ng_func_next_ent"] = float(-(p * np.log(p + 1e-12)).sum())
        out["ng_func_next_nuniq"] = len(nxt) / nf
    else:
        out["ng_func_next_ent"] = 0.0; out["ng_func_next_nuniq"] = 0.0
    # 内容字二元组 vs 含虚词二元组的比例
    out["ng_bg_content"] = float(np.mean([1.0 if (b[0] not in FUNC and b[1] not in FUNC) else 0.0 for b in bgs]))
    return out


def keys():
    L = ["\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b", "\u4e5d\u5341\u7532\u4e59\u4e19\u4e01\u620a\u5df1",
         "\u5e9a\u8f9b\u58ec\u7678\u5b50\u4e11\u5bc5\u536f", "\u8fb0\u5df3\u5348\u672a\u7533\u9149\u620c\u4ea5",
         "\u5b50\u4e11\u5bc5\u536f\u8fb0\u5df3\u5348\u672a", "\u7533\u9149\u620c\u4ea5\u4e00\u4e8c\u4e09\u56db"]
    r = feats_ngram(L)
    return sorted(r.keys()) if r else []


# ══════════════════════════════════════════════════════════════════
# §9.168 扩展：4-gram / 跨行搭配 / 虚词骨架搭配（16 维）
# ══════════════════════════════════════════════════════════════════
def feats_ngram2(lines):
    L = [x.strip() for x in lines if x and x.strip()]
    if len(L) < 6: return None
    cs = [c for c in "".join(L) if CJK(c)]
    if len(cs) < 20: return None
    R = _ref(); TOT = max(1, R["total"])
    out = {}
    def lfstats(ngrams, table, pref):
        if not ngrams: 
            out[pref+"_m"]=out[pref+"_p10"]=out[pref+"_unseen"]=out[pref+"_ent"]=0.0; return
        v = np.array([np.log((table.get(x, 0) + 0.5) / TOT) for x in ngrams])
        out[pref+"_m"] = float(v.mean()); out[pref+"_p10"] = float(np.percentile(v, 10))
        out[pref+"_unseen"] = float(np.mean([1.0 if x not in table else 0.0 for x in ngrams]))
        cnt = collections.Counter(ngrams)
        p = np.array(list(cnt.values()), float); p /= p.sum()
        out[pref+"_ent"] = float(-(p * np.log(p + 1e-12)).sum())
    # 4-gram
    qgs = [cs[i]+cs[i+1]+cs[i+2]+cs[i+3] for i in range(len(cs)-3)]
    lfstats(qgs, R.get("qg", {}), "ng2_qg")
    # 跨行（上一行末字 + 下一行首字）
    xls = []
    for a, b in zip(L, L[1:]):
        ca=[c for c in a if CJK(c)]; cb=[c for c in b if CJK(c)]
        if ca and cb: xls.append(ca[-1]+cb[0])
    lfstats(xls, R.get("xl", {}), "ng2_xl")
    # 虚词骨架
    sk = "".join(c if (c in FUNC or not CJK(c)) else "#" for c in "".join(L))
    sks = [sk[i:i+2] for i in range(len(sk)-1) if (sk[i] != "#" or sk[i+1] != "#")]
    lfstats(sks, R.get("sk", {}), "ng2_sk")
    # 骨架的重复结构
    csk = collections.Counter(sks)
    out["ng2_sk_rep"] = float(np.mean([1.0 if v > 1 else 0.0 for v in csk.values()])) if csk else 0.0
    out["ng2_qg_rep"] = float(np.mean([1.0 if v > 1 else 0.0 for v in collections.Counter(qgs).values()])) if qgs else 0.0
    out["ng2_xl_rep"] = float(np.mean([1.0 if v > 1 else 0.0 for v in collections.Counter(xls).values()])) if xls else 0.0
    return out


def keys2():
    L = ["\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b", "\u4e5d\u5341\u7532\u4e59\u4e19\u4e01\u620a\u5df1",
         "\u5e9a\u8f9b\u58ec\u7678\u5b50\u4e11\u5bc5\u536f", "\u8fb0\u5df3\u5348\u672a\u7533\u9149\u620c\u4ea5",
         "\u5b50\u4e11\u5bc5\u536f\u8fb0\u5df3\u5348\u672a", "\u7533\u9149\u620c\u4ea5\u4e00\u4e8c\u4e09\u56db"]
    r = feats_ngram2(L)
    return sorted(r.keys()) if r else []
