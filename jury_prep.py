# -*- coding: utf-8 -*-
"""§9.188 jury 预处理：① 去 markdown 痕迹 ② 自动按自然句分行
两项都在检测前完成，使「同一文本的不同粘贴形式」得到一致结果。
"""
import re

# ── markdown / 富文本痕迹
_MD_PATTERNS = [
    (re.compile(r"```[a-zA-Z]*\n?"), ""),          # 代码围栏
    (re.compile(r"~~~[a-zA-Z]*\n?"), ""),
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),  # 图片 → alt
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),   # 链接 → 文字
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),    # 标题
    (re.compile(r"^\s{0,3}>\s?", re.M), ""),         # 引用
    (re.compile(r"^\s{0,3}[-*+]\s+", re.M), ""),     # 无序列表
    (re.compile(r"^\s{0,3}\d+[.)]\s+", re.M), ""),   # 有序列表
    (re.compile(r"^\s{0,3}([-*_]\s*){3,}\s*$", re.M), ""),  # 分隔线
    (re.compile(r"\*\*\*(.+?)\*\*\*", re.S), r"\1"),
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),
    (re.compile(r"__(.+?)__", re.S), r"\1"),
    (re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", re.S), r"\1"),
    (re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", re.S), r"\1"),
    (re.compile(r"~~(.+?)~~", re.S), r"\1"),
    (re.compile(r"`([^`\n]+)`"), r"\1"),             # 行内代码
    (re.compile(r"^\s*\|.*\|\s*$", re.M), ""),       # 表格行
    (re.compile(r"^\s*:?-{3,}:?(\s*\|\s*:?-{3,}:?)+\s*$", re.M), ""),
    (re.compile(r"<[^>\n]{1,80}>"), ""),             # 简单 HTML 标签
    (re.compile(r"&nbsp;"), " "),
    (re.compile(r"&amp;"), "&"), (re.compile(r"&lt;"), "<"), (re.compile(r"&gt;"), ">"),
    (re.compile(r"^[\u200b\u200c\u200d\ufeff]+"), ""),  # 零宽字符
]
# 中文句末（用于自动分行）
_SENT_END = "。！？；!?;…"


def strip_markdown(text):
    """去除 markdown / HTML / 零宽字符等「粘贴痕迹」，保留文字内容"""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    for pat, rep in _MD_PATTERNS:
        s = pat.sub(rep, s)
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    s = re.sub(r"[ \t]+", " ", s)
    return s


def balanced_segment(lines, min_line=6, max_line=200, merge_min=6):
    """均衡分段：从句子级开始，若不够 min_line 行，才拆**最长**的句子（在逗号处）
       ⇒ 行既自然、又能凑够特征所需的行数
    """
    cur = [x for x in lines if x.strip()]
    guard = 0
    while len(cur) < min_line and guard < max_line:
        guard += 1
        # 找最长的行
        i = max(range(len(cur)), key=lambda k: len(cur[k])) if cur else -1
        if i < 0: break
        seg = cur[i]
        cuts = [q.strip() for q in re.split(r"(?<=[，,、：:])\s*", seg) if q.strip()]
        # 相邻小片合并，避免过碎
        merged = []
        for q in cuts:
            if merged and len(merged[-1]) < merge_min:
                merged[-1] += q
            else:
                merged.append(q)
        if len(merged) < 2:      # 无法再拆
            break
        cur = cur[:i] + merged + cur[i + 1:]
    return cur[:max_line]


def text_to_lines(text, min_line=6, max_line=200, long_line=40):
    """把任意粘贴形式变成「行列表」
       ① 已是多行 **且行短**（中位行长 ≤ long_line）⇒ 视为诗，直接用
       ② 否则先拆成自然句；不够 min_line 时再用 balanced_segment 细分
    """
    raw = [x.strip() for x in text.splitlines() if x.strip()]
    if len(raw) >= min_line:
        # ★★ 已有 ≥ min_line 行 ⇒ **完全保留原结构**
        #    理由：训练时的行结构就是推理时应有的结构；重新分行会破坏分布
        #    （只有不足 min_line 行时才需要自动分行）
        return raw[:max_line]
    parts = expand_sentences(raw if raw else [text])
    if len(parts) >= min_line:
        return parts[:max_line]
    if len(parts) >= 3:
        return balanced_segment(parts, min_line=min_line, max_line=max_line)
    # 句子都不到 3 句 ⇒ 直接均衡分段
    seg = balanced_segment(parts if parts else [text], min_line=min_line, max_line=max_line)
    if len(seg) >= 3:
        return seg
    # 最后兜底：按固定长度切
    flat = re.sub(r"\s+", "", text)
    if not flat: return []
    n = max(min_line, min(max_line, (len(flat) + min_line - 1) // min_line))
    k = (len(flat) + n - 1) // n
    return [flat[i * k:(i + 1) * k] for i in range(n) if flat[i * k:(i + 1) * k]]


def expand_sentences(lines):
    """把长段落拆成自然句（句末标点切分），保留短行原样"""
    out = []
    for ln in lines:
        if len(ln) <= 40:
            out.append(ln); continue
        ps = re.split(r"(?<=[" + re.escape(_SENT_END) + r"])\s*", ln)
        ps = [p.strip() for p in ps if p.strip()]
        out.extend(ps if ps else [ln])
    return out


def prepare(text, min_line=6, prefer_lines=None):
    """完整预处理：去 markdown → 自动分行
       prefer_lines=True 强制按行（诗）；False 强制拆句；None 自动
    """
    s = strip_markdown(text)
    if prefer_lines is False:
        return text_to_lines(s, min_line=min_line, long_line=0)
    return text_to_lines(s, min_line=min_line)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    T = """## 新能源的发展前景

**新能源**是指太阳能、风能、水能等可再生能源。近年来，随着技术进步，其成本大幅下降。太阳能光伏转换效率持续提升，海上风电集群加速并网，抽水蓄能成为调峰主力。

> 需要注意的是，新能源的间歇性仍是主要挑战。`储能技术`的突破将是关键。"""
    print("  去 markdown 后：")
    print("   ", repr(strip_markdown(T)[:120]))
    print("  自动分行（%d 行）：" % len(text_to_lines(strip_markdown(T), min_line=3)))
    for x in text_to_lines(strip_markdown(T), min_line=3): print("    %s" % x)
