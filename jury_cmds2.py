# -*- coding: utf-8 -*-
"""jury_cmds2.py — `analysis` / `research` 两个命令的实现

  analysis  用人话说出该文本最接近 AI / 人类的特征（不给概率）
            支持 score 的长文本模式（--seg 自动分段）与便捷模式（-f 多文件 / 管道）
  research  长文章深度研究
            -s 简单：按步长切块，逐块给体裁 + 人机判定
            -c 全面：滑窗（每次下移一行）读完全文后整合
                     主体裁 / 副体裁 / 平均 AI 概率 / 体裁切换点
"""
import io, os, sys

NL = chr(10)
CR = chr(13)


def _read(path):
    return io.open(path, encoding="utf-8", errors="replace").read()


def _lines(t):
    """把文本切成非空行（兼容 CRLF / LF）"""
    return [x.strip() for x in t.replace(CR, "").split(NL) if x.strip()]


# ── analysis ──
def cmd_analysis(a):
    import jury_diag as DG
    files = getattr(a, "file", None) or []
    top = getattr(a, "top", 4)
    seg = getattr(a, "seg", 0) or 0
    items = []
    if files:
        for f in files:
            if os.path.isdir(f):
                import glob
                for g in sorted(glob.glob(os.path.join(f, "*.txt"))):
                    items.append((g, _read(g)))
            elif os.path.exists(f):
                items.append((f, _read(f)))
            else:
                print("  skip (not found): %s" % f)
    if not items:
        if sys.stdin.isatty():
            print("  need -f FILE (or pipe text in)")
            return 2
        items.append(("<stdin>", sys.stdin.read()))
    for name, t in items:
        L = _lines(t)
        if not L:
            print("  (%s：空)" % name); continue
        if seg and len(L) > seg:
            print("  [长文模式：%d 行，按每段 %d 行分别诊断]" % (len(L), seg))
            for i in range(0, len(L), seg):
                part = L[i:i + seg]
                if len(part) < 6:
                    break
                print(DG.render(part, label="%s [%d-%d 行]" % (name, i + 1, i + len(part)), top=top))
                print()
        else:
            print(DG.render(L, label=name, top=top))
            print()
    return 0


# ── research ──
def cmd_research(a):
    import jury_research as JR
    p = getattr(a, "file", None)
    if isinstance(p, (list, tuple)):
        p = p[0] if p else None
    if not p:
        print("  need -f ARTICLE.txt"); return 2
    if not os.path.exists(p):
        print("  not found: %s" % p); return 2
    L = _lines(_read(p))
    if len(L) < 12:
        print("  article too short (%d lines, need >= 12)" % len(L)); return 2
    step = getattr(a, "step", 0) or None
    # ★ 手动步长校验：窗口至少要 6 行（检测器下限）
    if step is not None:
        if step < 6:
            print("  warn: step=%d is below the 6-line minimum; clamped to 6" % step)
            step = 6
        if step > len(L):
            print("  warn: step=%d exceeds article length (%d); clamped to %d" % (step, len(L), len(L)))
            step = len(L)
    if getattr(a, "comprehensive", False):
        _st = step if step else JR.auto_step(len(L))
        print("  comprehensive mode: sliding window (step=%d%s) ..." % (
            _st, "" if step else ", auto"))
        r = JR.comprehensive(L, step=step)
        print(JR.render_comprehensive(r, label=os.path.basename(p), lines=L,
                                     quote=bool(getattr(a, "quote", False))))
    else:
        r = JR.simple(L, step=step)
        print(JR.render_simple(r, label=os.path.basename(p), lines=L,
                              quote=bool(getattr(a, "quote", False))))
    return 0
