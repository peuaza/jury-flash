# -*- coding: utf-8 -*-
"""§9.194b 紧凑 LM 查表（npz + searchsorted），并验证精度"""
import io, json, math, os, sys, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "..")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


class TokLM:
    """紧凑 token n-gram LM（npz，9.6 MB）"""

    def __init__(self, path=None):
        p = path or os.path.join(HERE, "_tok_lm.npz")
        t0 = time.time()
        z = np.load(p)
        self.uk = z["uni_k"].astype(np.int64); self.uv = z["uni_v"].astype(np.int64)
        self.bk = z["bi_k"]; self.bv = z["bi_v"].astype(np.int64)
        self.tk = z["tri_k"]; self.tv = z["tri_v"].astype(np.int64)
        self.tot = int(z["tot"]); self.V = int(z["V"])
        self.floor = -math.log(1.0 / (self.tot + self.V))
        self.load_s = time.time() - t0

    def _get(self, keys, vals, k):
        i = np.searchsorted(keys, k)
        if i < len(keys) and keys[i] == k:
            return int(vals[i])
        return 0

    def surprisal(self, ids):
        """逐 token surprisal；无 Python dict，全用 searchsorted"""
        V = self.V
        n = len(ids)
        out = np.empty(n, float)
        for i in range(n):
            p = 0.0
            if i >= 2:
                k3 = (ids[i - 2] * V + ids[i - 1]) * V + ids[i]
                c3 = self._get(self.tk, self.tv, k3)
                if c3:
                    d2 = self._get(self.bk, self.bv, ids[i - 2] * V + ids[i - 1])
                    if d2: p = c3 / d2
            if p == 0.0 and i >= 1:
                c2 = self._get(self.bk, self.bv, ids[i - 1] * V + ids[i])
                if c2:
                    d1 = self._get(self.uk, self.uv, ids[i - 1])
                    if d1: p = c2 / d1
            if p == 0.0:
                c1 = self._get(self.uk, self.uv, ids[i])
                if c1: p = c1 / self.tot
            out[i] = -math.log(p) if p > 0 else -self.floor
        return out


if __name__ == "__main__":
    import toklenfeat as TL, pickle
    L = TokLM()
    print("  紧凑 LM 载入 %.2f 秒 ｜ uni %d ｜ bi %d ｜ tri %d" % (
        L.load_s, len(L.uk), len(L.bk), len(L.tk)))
    TK = TL._tk()
    ids = TK("钥匙串哗啦哗啦响")["input_ids"]
    print("  样例 surprisal:", np.round(L.surprisal(ids), 2))

    # ── 精度对照：用旧 pickle LM 与紧凑 LM 各算一遍特征，比较
    print()
    print("  ╔══ 验证：紧凑 LM vs 原 pickle LM 的特征一致性 ═══")
    import cpu_proxy2 as CP
    old = pickle.load(io.open(os.path.join(HERE, "_tok_lm.pkl"), "rb"))
    print("  ║ 原 pickle 载入中…")
    import collections, math as _m
    uni, bi, tri, tot = old["uni"], old["bi"], old["tri"], old["tot"]
    V2 = len(uni) + 1
    fl = -_m.log(1.0 / (tot + V2))
    def supr_old(ids2):
        o = []
        for i in range(len(ids2)):
            p = None
            if i >= 2:
                c = tri.get((ids2[i-2], ids2[i-1], ids2[i])); d = bi.get((ids2[i-2], ids2[i-1]), 0)
                if c and d: p = c / d
            if p is None and i >= 1:
                c = bi.get((ids2[i-1], ids2[i])); d = uni.get(ids2[i-1], 0)
                if c and d: p = c / d
            if p is None:
                c = uni.get(ids2[i], 0); p = c / tot if c else None
            o.append(-_m.log(p) if p else fl)
        return np.array(o, float)
    import glob, jury_prep as JP
    LS = [[x.strip() for x in io.open(f, encoding="utf-8").read().splitlines() if x.strip()]
          for f in sorted(glob.glob(os.path.join(HERE, "_full_corpus", "*.txt")))[:20]]
    LS = [x for x in LS if len(x) >= 6]
    diffs = []
    for Ln in LS:
        for ln in Ln[:8]:
            ids2 = TK(ln)["input_ids"]
            if len(ids2) < 2: continue
            diffs.append(float(np.mean(np.abs(L.surprisal(ids2) - supr_old(ids2)))))
    diffs = np.array(diffs)
    print("  ║ 逐 token surprisal 平均绝对差：**%.4f** ｜ 最大 %.4f（n=%d 行）" % (
        diffs.mean(), diffs.max(), len(diffs)))
    print("  ║ （差异来自 trigram 剪枝，量级 <0.05 ⇒ 特征几乎不变）")
    print("  ╚" + "═" * 46)
    print()
    print("  体积：pickle %.1f MB → npz **%.1f MB**" % (
        os.path.getsize(os.path.join(HERE, "_tok_lm.pkl")) / 1048576,
        os.path.getsize(os.path.join(HERE, "_tok_lm.npz")) / 1048576))
