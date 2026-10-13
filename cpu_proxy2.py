# -*- coding: utf-8 -*-
"""§9.193 B：更强的 CPU 代理
   ① **token 级（BPE）n-gram LM** —— 用 Qwen 分词器，更接近 GPU 的语义单元
   ② **压缩率族** —— 逐行 gzip/zlib 压缩比（可预测性的经典代理）
   检验：能否在「用户诗 vs 同族473」上超过字符 n-gram（0.57~0.63）
"""
import collections, gzip, io, json, math, os, pickle, sys, time, zlib
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "..")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import toklenfeat as TL, jury_prep as JP

HERE = os.path.dirname(os.path.abspath(__file__))
LM2 = os.path.join(HERE, "_tok_lm.pkl")
SRC = r"C:\Users\shini\poetry_axis\corpus_modern\modern_poems.jsonl"


# ══════════════ ① token 级 n-gram LM ══════════════
def build_tok_lm(maxl=60000):
    TK = TL._tk()
    uni = collections.Counter(); bi = collections.Counter(); tri = collections.Counter()
    n = 0; t0 = time.time()
    for l in io.open(SRC, encoding="utf-8"):
        try: r = json.loads(l)
        except Exception: continue
        L = [x.strip() for x in (r.get("lines") or []) if x.strip()]
        if len(L) < 6: continue
        ids = []
        for ln in L:
            try: ids += TK(ln)["input_ids"]
            except Exception: pass
        if len(ids) < 15: continue
        uni.update(ids)
        bi.update(zip(ids, ids[1:]))
        tri.update(zip(ids, ids[1:], ids[2:]))
        n += 1
        if n >= maxl: break
        if n % 15000 == 0: print("      %d 首 ｜ %.1f 分" % (n, (time.time() - t0) / 60), flush=True)
    print("    token LM：语料 %d 首 ｜ uni %d ｜ bi %d ｜ tri %d" % (n, len(uni), len(bi), len(tri)))
    return {"uni": uni, "bi": bi, "tri": tri, "tot": sum(uni.values())}


_TLM = {}
def _tk():
    """tokenizer 单例"""
    if "tk" not in _TK_CACHE:
        import toklenfeat as _TL
        _TK_CACHE["tk"] = _TL._tk()
    return _TK_CACHE["tk"]


def load_lm(path=None):
    """加载**紧凑 npz** token LM（9.6 MB，载入 0.09 秒）"""
    from toklm import TokLM
    p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tok_lm.npz")
    _TLM["lm"] = TokLM(p)
    return _TLM["lm"]


def tok_supr(ids):
    """逐 token surprisal（**紧凑 npz LM**，searchsorted 查表）"""
    return _TLM["lm"].surprisal(list(ids))


def feats_tok_lm(lines, tk):
    L = [x.strip() for x in lines if x and x.strip()][:16]
    if len(L) < 6: return None
    alls, isd, imx, isp = [], [], [], []
    for ln in L:
        try: ids = tk(ln)["input_ids"]
        except Exception: continue
        if len(ids) < 2: continue
        v = tok_supr(ids)
        alls.append(v)
        isd.append(float(v.std())); imx.append(float(v.max())); isp.append(float(v.max() - v.min()))
    if not alls: return None
    S = np.concatenate(alls); q = np.percentile(S, [10, 25, 50, 75, 90, 95, 99])
    return {"tl_mean": float(S.mean()), "tl_sd": float(S.std()), "tl_max": float(S.max()),
            "tl_range": float(S.max() - S.min()), "tl_p10": float(q[0]), "tl_p25": float(q[1]),
            "tl_p50": float(q[2]), "tl_p75": float(q[3]), "tl_p90": float(q[4]),
            "tl_p95": float(q[5]), "tl_p99": float(q[6]), "tl_iqr": float(q[3] - q[1]),
            "tl_hi_frac": float((S > 8).mean()), "tl_lo_frac": float((S < 1.5).mean()),
            "tl_top5": float(np.sort(S)[-max(1, len(S) // 20):].mean()),
            "tl_skew": float(((S - S.mean()) ** 3).mean() / (S.std() ** 3 + 1e-9)),
            "tl_kurt": float(((S - S.mean()) ** 4).mean() / (S.std() ** 4 + 1e-9)),
            "tl_imax_m": float(np.mean(imx)), "tl_imax_max": float(np.max(imx)),
            "tl_ispan_m": float(np.mean(isp)), "tl_isd_m": float(np.mean(isd)),
            "tl_isd_sd": float(np.std(isd))}


# ══════════════ ② 压缩率族 ══════════════
def _cr(s, lvl=9):
    b = s.encode("utf-8")
    if not b: return 0.0
    return len(gzip.compress(b, lvl)) / len(b)


# ══════════════ ①b 顺序 LM 特征（★ 跨行上下文）══════════════
def feats_seq_lm(lines):
    """整篇 tokenize（保留跨行上下文）⇒ surprisal 序列的**时序结构**特征
       ★ 现有 tl_* 是**逐行独立**算的；本族看的是跨行/时序结构，
         攻击方很难在保持可读性的同时匹配（§9.230 原型 AUC 0.063~0.28）"""
    L = [x.strip() for x in lines if x and str(x).strip()]
    if len(L) < 6: return None
    text = chr(10).join(L)
    try:
        ids = _tk()(text)["input_ids"]
    except Exception:
        return None
    if len(ids) < 20: return None
    try:
        v = np.asarray(tok_supr(ids), float)
    except Exception:
        return None
    n = len(v)
    if n < 20: return None
    # 逐行起始位置的 surprisal（跨行转移处）
    starts = []; pos = 0
    for ln in L:
        try: k = len(_tk()(ln)["input_ids"])
        except Exception: k = 0
        if pos < n: starts.append(pos)
        pos += k + 1
    sv = v[np.array([x for x in starts if x < n], int)] if starts else np.array([v[0]])
    # 线性趋势
    xx = np.arange(n, dtype=np.float64); xx = xx - xx.mean()
    slope = float((xx * (v - v.mean())).sum() / ((xx * xx).sum() + 1e-9))
    # 自相关
    vc = v - v.mean(); den = float((vc * vc).sum()) + 1e-9
    ac = []
    for lag in (1, 2, 3, 5):
        ac.append(float((vc[:-lag] * vc[lag:]).sum() / den) if n > lag else 0.0)
    # 前后三分之一差
    t3 = n // 3
    d13 = float(v[t3:2 * t3].mean() - v[:t3].mean()) if t3 > 0 else 0.0
    # 分布熵（20 桶）
    h, _ = np.histogram(v, bins=20)
    p = h / max(1, h.sum()); p = p[p > 0]
    ent = float(-(p * np.log(p)).sum())
    burst = float((v > v.mean() + 2 * v.std()).mean())
    return {"sl_mean": float(v.mean()), "sl_sd": float(v.std()),
            "sl_slope": slope, "sl_ac1": ac[0], "sl_ac2": ac[1],
            "sl_ac3": ac[2], "sl_ac5": ac[3], "sl_d13": d13,
            "sl_ent": ent, "sl_burst": burst,
            "sl_ls_m": float(sv.mean()), "sl_ls_sd": float(sv.std()),
            "sl_ls_max": float(sv.max()), "sl_n": float(n),
            "sl_per_line": float(n / max(1, len(L)))}


_TK_CACHE = {}


def feats_comp(lines):
    L = [x.strip() for x in lines if x and x.strip()][:24]
    if len(L) < 6: return None
    crs = [_cr(x) for x in L if len(x) >= 2]
    if len(crs) < 4: return None
    a = np.array(crs, float)
    # 全篇（跨行）压缩率：越高 ⇒ 越不可压缩 ⇒ 越不可预测
    doc = "\n".join(L)
    return {"cz_line_m": float(a.mean()), "cz_line_sd": float(a.std()),
            "cz_line_max": float(a.max()), "cz_line_min": float(a.min()),
            "cz_doc": float(_cr(doc)), "cz_char_ent": float(_char_ent(doc)),
            "cz_bg_rep": float(_rep(doc, 2)), "cz_tg_rep": float(_rep(doc, 3)),
            "cz_line_range": float(a.max() - a.min()),
            "cz_cv": float(a.std() / (a.mean() + 1e-9))}


def _char_ent(s):
    s = "".join(c for c in s if "\u4e00" <= c <= "\u9fff")
    if not s: return 0.0
    c = collections.Counter(s); n = len(s)
    return float(-sum((v / n) * math.log(v / n) for v in c.values()))


def _rep(s, k):
    g = [s[i:i + k] for i in range(len(s) - k + 1)]
    return 1 - len(set(g)) / max(1, len(g))


# ══════════════ 主 ══════════════
if __name__ == "__main__":
    TK = TL._tk()
    TLM = load_lm()
    print("    已载入紧凑 token LM（uni %d ｜ bi %d ｜ tri %d ｜ 载入 %.2f 秒）" % (
        len(TLM.uk), len(TLM.bk), len(TLM.tk), TLM.load_s))

    def load(f):
        if not os.path.exists(f): return []
        if f.endswith(".json"):
            t = io.open(f, encoding="utf-8").read().strip()
            try:
                d = json.loads(t); return d if isinstance(d, list) else [d]
            except Exception: return []
        return [json.loads(l) for l in io.open(f, encoding="utf-8") if l.strip()]

    import glob as _g
    LINES = {}
    LINES["FC"] = [[x.strip() for x in io.open(f, encoding="utf-8").read().splitlines() if x.strip()]
                   for f in sorted(_g.glob("_full_corpus/*.txt"))]
    LINES["FC"] = [L for L in LINES["FC"] if len(L) >= 6]
    LINES["H1"] = [r["lines"] for r in load("../corpus_web/human1000.json") if r.get("lines")]
    LINES["NP"] = [r["lines"] for r in load("ai_neutral_poem.jsonl") if r.get("lines")]
    LINES["AP"] = [r["lines"] for r in load("ai_poems.jsonl") if r.get("lines")]
    UP = r"C:\Users\shini\poetry_axis\external_call\poem_reading\_poet_corpus"
    LINES["U"] = [[x.strip() for x in io.open(f, encoding="utf-8", errors="replace").read().splitlines()
                   if x.strip()] for f in sorted(_g.glob(os.path.join(UP, "*.txt")))]
    LINES["U"] = [L for L in LINES["U"] if len(L) >= 6]
    LINES["EX"] = [r["lines"] for r in load("jury_samples/human_filtered.jsonl") if r.get("lines")]
    _G = []
    for _f in ("poet_passed.jsonl", "poet_passed2.jsonl", "poet_passed3.jsonl",
               "poet_passed4.jsonl", "poet_passed5.jsonl"):
        _G += [r["lines"] for r in load(_f) if r.get("lines")]
    LINES["G"] = _G
    MS = [json.loads(l) for l in io.open("_modern_feats.jsonl", encoding="utf-8")]
    LINES["MD"] = [r["lines"] for r in MS if r.get("p", 1) < 0.5 and len(r.get("lines") or []) >= 6][:250]
    LINES["MX"] = [r["lines"] for r in MS if r.get("p", 0) >= 0.8579 and len(r.get("lines") or []) >= 6][:250]

    OUTF = {}
    for g, LS in LINES.items():
        rows = []
        t0 = time.time()
        for L in LS:
            a = feats_tok_lm(L, TK); b = feats_comp(L)
            rows.append({**a, **b} if (a and b) else None)
        OUTF[g] = rows
        print("    %-3s %4d ｜ %.1f 分" % (g, len(LS), (time.time() - t0) / 60), flush=True)
    pickle.dump(OUTF, io.open("_cpu_proxy2.json", "wb"), protocol=4)

    K2 = sorted([k for k in OUTF["FC"][0] if k])
    def arr(g):
        return np.array([[v[k] for k in K2] for v in OUTF[g] if v])
    def auc(a, b):
        allv = np.r_[a, b]; r = np.argsort(np.argsort(allv))
        return r[:len(a)].mean() / (len(allv) - 1)
    FC, H1, U = arr("FC"), arr("H1"), arr("U")
    print()
    print("  ╔══ ★★★★ 单维 AUC 对比（top 各 8）═══")
    print("  ║ ── 同族473 vs human1000 ──")
    sc = [(max(auc(FC[:, j], H1[:, j]), 1 - auc(FC[:, j], H1[:, j])), K2[j]) for j in range(len(K2))]
    for a, k in sorted(sc, reverse=True)[:8]: print("  ║ %-14s %.3f" % (k, a))
    print("  ║ ── ★ 用户诗 vs 同族473（越高越能救）──")
    n = min(len(U), len(FC))
    sc2 = [(max(auc(U[:n, j], FC[:n, j]), 1 - auc(U[:n, j], FC[:n, j])), K2[j]) for j in range(len(K2))]
    for a, k in sorted(sc2, reverse=True)[:8]: print("  ║ %-14s %.3f" % (k, a))
    print("  ╚" + "═" * 36)
    print()
    print("  对照：字符 n-gram 在「用户 vs 同族」上是 0.57~0.63；GPU surprisal 是 0.63~0.68")
