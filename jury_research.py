# -*- coding: utf-8 -*-
"""jury_research.py — 长文章深度研究（切片分析）

两种模式
--------
  -s 简单模式：把文章按步长**切块**（不重叠），逐块给出「体裁 + 人机判定」
  -c 全面模式：**滑窗**从第一行开始、每次下移一行，读完全文后**整合信息**：
                主体裁（出现最多的）｜ 副体裁（与主体裁不同的）｜ 平均 AI 概率
               还给出「体裁切换分布」与「AI 概率沿文章的变化」

默认步长：自动（约 1/6 文本长度，限制在 8~40 行）
输出：表格
"""
import io, os, sys
import numpy as np


def _cut(t, n=26):
    t = str(t).replace(chr(10), " ").strip()
    return t if len(t) <= n else t[:n] + "…"


def _q_line(L, ln, n=26):
    """第 ln 行（1-based）的开头"""
    if 1 <= ln <= len(L):
        return _cut(L[ln - 1], n)
    return ""


def _q_win(L, a, b, n=22):
    """窗口 [a,b]（1-based，含端点）的首尾"""
    h = _q_line(L, a, n)
    t = _q_line(L, b, n) if b != a else ""
    return h, t


def _det():
    import jury_det as JD
    return JD.Detector(model="auto_c"), JD.Detector(model="auto_p")


def _score_one(L, Dc, Dp):
    """返回 (体裁, 人机, AI概率, 说明)"""
    L = [x for x in L if str(x).strip()]
    if len(L) < 6:
        return (None, None, None, "行数不足", None)
    try:
        oc = Dc.predict_lines(L)
        tf = oc.get("top_feats") or {}
        g = tf.get("判定体裁")
        if not oc.get("ok"):
            return (None, None, None, "判分失败", None)
        p_ai = float(tf.get("AI类概率") if tf.get("AI类概率") is not None else (tf.get("该类概率") or 0.0))
        p_top = float(tf.get("该类概率") or 0.0)
        is_ai = bool(oc.get("is_ai"))
        return (g, ("AI" if is_ai else "人类"), p_ai, None, p_top)
    except Exception as e:
        return (None, None, None, str(e)[:30], None)


def auto_step(n):
    s = int(round(n / 6.0))
    return max(8, min(40, s))


def simple(lines, step=None, quiet=False):
    """切块（不重叠）⇒ 每块体裁/人机"""
    L = [x for x in lines if str(x).strip()]
    n = len(L)
    if step is None:
        step = auto_step(n)
    Dc, Dp = _det()
    rows = []
    for i in range(0, n, step):
        seg = L[i:i + step]
        if len(seg) < 6:
            break
        g, v, p, err, pt = _score_one(seg, Dc, Dp)
        rows.append({"from": i + 1, "to": i + len(seg), "n": len(seg),
                     "genre": g, "verdict": v, "p_ai": p, "p_top": pt, "err": err})
    return {"mode": "simple", "n": n, "step": step, "rows": rows}


def comprehensive(lines, step=None, quiet=False):
    """滑窗（每次下移一行）⇒ 整合：主体裁 / 副体裁 / 平均 AI 概率"""
    L = [x for x in lines if str(x).strip()]
    n = len(L)
    if step is None:
        step = auto_step(n)
    Dc, Dp = _det()
    rows = []
    for i in range(0, n - 6 + 1):
        seg = L[i:i + step]
        if len(seg) < 6:
            break
        g, v, p, err, pt = _score_one(seg, Dc, Dp)
        rows.append({"from": i + 1, "to": min(i + step, n), "genre": g,
                     "verdict": v, "p_ai": p, "p_top": pt, "err": err})
        if not quiet and i % 10 == 0:
            print("    ...已读 %d/%d 行" % (i + 1, n), flush=True)
    ok = [r for r in rows if r["genre"]]
    from collections import Counter
    gc = Counter(r["genre"] for r in ok)
    main = gc.most_common(1)[0][0] if gc else None
    sec = [(g, c) for g, c in gc.most_common() if g != main]
    # ★ 多种聚合口径（显式列出，避免歧义）
    ps = [r["p_ai"] for r in ok if r.get("p_ai") is not None]
    p_ai = float(np.mean(ps)) if ps else 0.0                       # (a) 按窗口等权
    _sp = sum(ps)
    p_ai_cw = float(sum(x * x for x in ps) / _sp) if _sp > 1e-9 else 0.0   # (b) 置信度加权 Σp²/Σp
    # (c) 按行等权：先算每行的均值，再对行求平均
    _pl = {}
    for r in ok:
        if r.get("p_ai") is None:
            continue
        for ln in range(r["from"], r["to"] + 1):
            _pl.setdefault(ln, []).append(r["p_ai"])
    p_ai_lw = float(np.mean([np.mean(v) for v in _pl.values()])) if _pl else 0.0
    # (d) 判 AI 的比例（verdict 口径）
    ai_ratio_v = float(np.mean([1.0 if r["verdict"] == "AI" else 0.0 for r in ok])) if ok else 0.0
    ai_ratio = (sum(1 for r in ok if r["verdict"] == "AI") / len(ok)) if ok else 0.0
    # 体裁切换点
    # ★ 先合并「短抖动」：把连续 <RUN 个窗口的体裁视为噪声并入邻段
    RUN = 3
    seq = [(r["from"], r["genre"]) for r in ok]
    def _runs(seq):
        out = []; i = 0
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and seq[j + 1][1] == seq[i][1]:
                j += 1
            out.append([seq[i][0], seq[j][0], seq[i][1], j - i + 1])
            i = j + 1
        return out
    segs = _runs(seq)
    changed = True
    while changed and len(segs) > 2:
        changed = False
        for k in range(len(segs)):
            if segs[k][3] < RUN:
                # 并入前一段（若有）否则并入后一段
                tgt = k - 1 if k > 0 else k + 1
                segs[tgt][1] = max(segs[tgt][1], segs[k][1])
                if tgt > k:
                    segs[tgt][0] = min(segs[tgt][0], segs[k][0])
                segs.pop(k); changed = True; break
    # 合并后可能出现相邻同体裁（噪声段被并入）⇒ 若同体裁则合并
    merged = []
    for a, b, g, ln in segs:
        if merged and merged[-1][2] == g:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][3] += ln
        else:
            merged.append([a, b, g, ln])
    segs = merged
    switches = []
    for k in range(1, len(segs)):
        if segs[k][2] != segs[k - 1][2]:
            switches.append((segs[k][0], segs[k - 1][2], segs[k][2]))
    # ★ 逐行聚合：每行收集所有覆盖它的窗口读数
    from collections import Counter as _C
    per_line = []
    for ln in range(1, n + 1):
        cov = [r for r in rows if r["from"] <= ln <= r["to"] and r["genre"]]
        if not cov:
            per_line.append({"line": ln, "n_read": 0})
            continue
        gc2 = _C(r["genre"] for r in cov)
        mg = gc2.most_common(1)[0][0]
        secs = [(g, c) for g, c in gc2.most_common() if g != mg]
        # 加权 AI 估计：以每个窗口的 AI 类总概率为权重（再归一）
        wsum = sum((r.get("p_ai") or 0.0) for r in cov)
        wgt = sum(1 for r in cov if (r.get("p_ai") or 0.0) >= 0.5)
        # 简单平均 + 概率加权平均
        p_mean = float(np.mean([r.get("p_ai") or 0.0 for r in cov]))
        p_w = float(np.sum([(r.get("p_ai") or 0.0) ** 2 for r in cov]) / max(1e-9, wsum))
        ai_ratio_l = float(np.mean([1.0 if r["verdict"] == "AI" else 0.0 for r in cov]))
        vc = _C(r["verdict"] for r in cov if r.get("verdict"))
        mv = vc.most_common(1)[0][0] if vc else "—"
        per_line.append({"line": ln, "n_read": len(cov), "main": mg, "sec": secs,
                         "p_mean": p_mean, "p_w": p_w, "ai_ratio": ai_ratio_l,
                         "verdict": mv, "vc": dict(vc)})
    # ★ 整体判定：按行统计各行的「众数判定」，再取多数
    from collections import Counter as _C2
    vlines = _C2(r.get("verdict") for r in per_line if r.get("n_read") and r.get("verdict"))
    n_ai = vlines.get("AI", 0); n_hu = vlines.get("人类", 0); n_un = vlines.get("不确定", 0)
    tot_v = n_ai + n_hu + n_un
    if tot_v == 0:
        overall = "无法判定"
    else:
        frac_ai = n_ai / tot_v
        if frac_ai >= 0.80:
            overall = "AI"
        elif frac_ai <= 0.20:
            overall = "人类"
        else:
            overall = "混合（不确定）"
    return {"mode": "comprehensive", "n": n, "step": step, "rows": rows,
            "overall": overall, "n_ai_lines": n_ai, "n_hu_lines": n_hu,
            "n_un_lines": n_un, "n_vlines": tot_v,
            "main": main, "sec": sec, "p_ai": p_ai, "ai_ratio": ai_ratio,
            "switches": switches, "segments": segs, "n_windows": len(ok),
            "per_line": per_line, "p_ai_cw": p_ai_cw, "p_ai_lw": p_ai_lw,
            "ai_ratio_v": ai_ratio_v, "n_lines_covered": len(_pl)}


# ── 渲染 ──
def render_simple(res, label=None, lines=None, quote=False):
    out = []
    out.append("=" * 74)
    if label:
        out.append("  文章：%s" % label)
    out.append("  模式：简单切块（不重叠）｜ 共 %d 行 ｜ 步长 %d 行 ｜ %d 片" % (
        res["n"], res["step"], len(res["rows"])))
    out.append("-" * 74)
    if quote:
        out.append("  %-12s %-5s %-8s %-6s %-10s %s" % ("行范围", "行数", "体裁", "判定", "AI类总概率", "该类概率"))
    else:
        out.append("  %-12s %-5s %-8s %-6s %-10s %s" % ("行范围", "行数", "体裁", "判定", "AI类总概率", "该类概率"))
    out.append("-" * 74)
    for r in res["rows"]:
        if r["err"]:
            out.append("  %-12s %-5d %-8s %-6s %-10s %s" % (
                "%d-%d" % (r["from"], r["to"]), r["n"], "—", "—", "—", "（%s）" % r["err"]))
        else:
            out.append("  %-12s %-5d %-8s %-6s %-10.3f %.3f" % (
                "%d-%d" % (r["from"], r["to"]), r["n"], r["genre"], r["verdict"],
                r["p_ai"], r.get("p_top") or 0.0))
        if quote and lines:
            h, t = _q_win(lines, r["from"], r["to"])
            out.append("        ↳ 首：%s" % h)
            if t:
                out.append("        ↳ 尾：%s" % t)
    out.append("-" * 74)
    return "\n".join(out)


def render_comprehensive(res, label=None, lines=None, quote=False):
    from collections import Counter
    out = []
    ok = [r for r in res["rows"] if r["genre"]]
    out.append("=" * 74)
    if label:
        out.append("  文章：%s" % label)
    out.append("  模式：全面滑窗（每次下移一行）｜ 共 %d 行 ｜ 窗口 %d 行 ｜ %d 个窗口" % (
        res["n"], res["step"], res["n_windows"]))
    out.append("")
    out.append("  ── 整合结论 ──")
    _ov = res.get("overall", "?")
    _mark = {"AI": "**AI**", "人类": "**人类**", "混合（不确定）": "**混合（不确定）**"}.get(_ov, _ov)
    out.append("    ★ 整体判定  ：%s   ← 按行统计各行的众数判定后取多数" % _mark)
    out.append("       判 AI 的行 %d ｜ 判人类的行 %d ｜ 不确定 %d（共 %d 行有读数）" % (
        res.get("n_ai_lines", 0), res.get("n_hu_lines", 0),
        res.get("n_un_lines", 0), res.get("n_vlines", 0)))
    out.append("       （规则：AI 行占比 ≥80% ⇒ AI；≤20% ⇒ 人类；中间 ⇒ 混合）")
    out.append("")
    out.append("    主体裁      ：**%s**（%d/%d 个窗口）" % (
        res["main"], (Counter(r["genre"] for r in ok).get(res["main"], 0)), len(ok)))
    if res["sec"]:
        out.append("    副体裁      ：%s" % "、".join(
            "%s(%d)" % (g, c) for g, c in res["sec"][:4]))
    else:
        out.append("    副体裁      ：无（全文体裁一致）")
    out.append("    判 AI 的窗口占比 ：**%.0f%%**（%d/%d）  ← 与「整体判定」一致的口径" % (
        100 * res.get("ai_ratio_v", 0), int(round(res.get("ai_ratio_v", 0) * len(ok))), len(ok)))
    out.append("")
    out.append("    ⚠ 「倾向分」仅供参考，**不是概率、不可与 0.70 门槛比较**（判定看的是每行的众数判定）：")
    out.append("        按窗口等权   %.3f ｜ 置信度加权  %.3f ｜ 按行等权  %.3f" % (
        res["p_ai"], res.get("p_ai_cw", 0), res.get("p_ai_lw", 0)))
    out.append("        倾向分 = 各窗口 14 类 softmax 里 **7 个 AI 类之和**")
    out.append("        它偏高不等于判 AI：若 AI 类分散（0.6+0.3+0.09=0.99）而无一个过保守门槛 0.70，")
    out.append("        该行仍判「人类」⇒ 所以「倾向分 0.73 却整体判人类」是**正常现象**")
    if res["switches"]:
        out.append("    体裁切换点  ：%d 处" % len(res["switches"]))
        for pos, a, b in res["switches"][:8]:
            out.append("        第 %d 行起：%s → %s%s" % (
                pos, a, b, ("    ｜ %s" % _q_line(lines, pos, 30)) if (quote and lines) else ""))
        # 主要段落
        segs = res.get("segments") or []
        if segs:
            out.append("    体裁段落    ：")
            for a, b, g, ln in segs[:8]:
                out.append("        第 %d-%d 行：%s（%d 个窗口）" % (a, b, g, ln))
    out.append("")
    out.append("  ── 逐行整合表（每行聚合同一行的全部读数）──")
    out.append("  %-6s %-8s %-12s %-16s %-12s %-10s %-8s %s" % (
        "行", "主体裁", "判定", "副体裁", "AI倾向(加权)", "AI倾向(均值)", "判AI比例", "读数次数"))
    pl = res.get("per_line") or []
    # ★ 触发新行：体裁、副体裁、**人机判定** 任一变化
    prev_key = None
    for r in pl:
        if not r.get("n_read"):
            continue
        key = (r["main"], tuple(g for g, _ in r["sec"]), r.get("verdict"))
        if key != prev_key:
            secs = "、".join("%s(%d)" % (g, c) for g, c in r["sec"][:3]) or "—"
            out.append("  %-6d %-8s %-12s %-16s %-12.3f %-10.3f %-8s %d" % (
                r["line"], r["main"], r.get("verdict") or "—", secs, r["p_w"], r["p_mean"],
                "%.0f%%" % (100 * r["ai_ratio"]), r["n_read"]))
            if quote and lines:
                out.append("         ↳ %s" % _q_line(lines, r["line"], 40))
            prev_key = key
    out.append("  （只显示**体裁/副体裁/人机判定**发生变化的行；`AI倾向` 为参考分（非概率）｜`判AI比例` = 判 AI 的读数占比（与判定一致））")
    out.append("")
    out.append("  ── 逐窗口明细（每 10 个窗口抽 1 行显示）──")
    out.append("  %-10s %-8s %-6s %-10s %s" % ("起始行", "体裁", "判定", "AI类总概率", "该类概率"))
    for i, r in enumerate(ok):
        if i % 10 == 0 or i == len(ok) - 1:
            out.append("  %-10s %-8s %-6s %-10.3f %.3f" % (
                r["from"], r["genre"], r["verdict"], r["p_ai"], r.get("p_top") or 0.0))
            if quote and lines:
                h, t = _q_win(lines, r["from"], r["to"])
                out.append("      ↳ 首：%s" % h)
                if t:
                    out.append("      ↳ 尾：%s" % t)
    out.append("-" * 74)
    return "\n".join(out)
