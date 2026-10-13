# -*- coding: utf-8 -*-
"""jury_diag.py — 分析模式：用**人话**说出该文本最接近 AI / 人类的特征

   参照 §9.61「评审式反馈」：**给人话诊断，不给概率**
   原理：
     ① 算 145 维特征
     ② 与该体裁的「人类类」「AI 类」参考分布比较
          偏向 = (x-μ_H)/σ_H − (x-μ_A)/σ_A
          >0 ⇒ 这一维更像 AI ／ <0 ⇒ 更像人类
     ③ 两侧各取最显著的若干组，用 HINTS 翻成**人话 + 可操作建议**
"""
import io, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# 特征组 → (像 AI 时的现象, 可操作建议, 像人类时的正面表述)
HINTS = [
    (("tk_ttr", "vo_mattr_m", "tk_entropy", "tk_top1", "vo_entropy", "if_ttr",
      "tk_hapax", "vo_zipf", "vo_hhi", "if_ttr_slope", "vo_mattr_slope"),
     "用词**重复、缺乏变化**：同一个词反复出现，词汇量增长太平滑，频次分布太「标准」",
     "多换具体的名词，少重复同一个词；可以用一次就丢的生僻说法",
     "用词**变化丰富**：同一个词很少重复，频次分布不规整（有个人偏好）"),

    (("ls_clause_m", "ls_clause_sd", "ls_clauselen_m", "ls_clauselen_sd",
      "cz_line_sd", "cz_line_range", "tk_linelen_sd", "ls_len_alt", "cz_cv",
      "tk_lineend_sd", "sf_skel_len_sd"),
     "**句子长度太均匀**：每句都差不多长，节奏像机器排的",
     "刻意打乱长短：插一句极短的（3~5 字），再来一句很长的；避免工整对仗",
     "**句子长短错落**：节奏忽长忽短，不像机器排的"),

    (("sf_func_density", "ng_func_next_ent", "ng_func_next_nuniq", "sf_skel_rep",
      "sf_skel_uniq", "sf_abstract", "sf_conc_minus_abs", "sf_field_dom", "sf_field_ent"),
     "**虚词／抽象词偏多**：「的、着、了、和、以及」和概念大词堆得比较密",
     "删掉一半的「的/了/和/以及」；把「意义/价值/温度/力量」换成能看见、能摸到的东西",
     "**虚词／抽象词克制**：不靠「的、了、和」堆句子，具体词多"),

    (("cz_bg_rep", "cz_tg_rep", "ng_bg_rep", "ng_tg_rep", "ls_head_rep", "ls_tail_rep",
      "ls_head_uniq", "ls_tail_uniq", "if_rep_lag1", "if_rep_lag2", "if_rep_lag3",
      "if_rep_ratio", "vo_rep_near", "vo_rep_cross", "vo_repdist_m", "vo_repdist_sd",
      "vo_repdist_med"),
     "**重复的词语搭配偏多**：相邻句子结构或用词高度相似，回扣太规律",
     "相邻两句不要用同一个开头、同一句式；打破「隔几句就回扣一次」的规律",
     "**搭配有变化**：相邻句子结构和用词差异大，回扣不规律"),

    (("ng_bg_unseen", "ng_bg_rare", "ng_bg_common", "ng_bg_content", "ng_bg_lf_m",
      "ng_bg_lf_p10", "ng_bg_lf_p90", "ng_bg_lf_sd", "ng_tg_unseen"),
     "**用词都在常见范围内**：几乎看不到冷僻词、专有名词、意外搭配",
     "加入生僻/方言/专有名词/具体型号，让一些搭配出乎意料",
     "**有冷僻词／意外搭配**：会出现生僻词、专有名词、跳出常见范围的组合"),

    (("sf_comma_share", "sf_period_share", "sf_npunct", "tk_lineend_punct",
      "sf_punct_ent", "sf_punct_per_line_m", "sf_punct_per_line_sd",
      "sf_punct_uniq_per_line", "tk_punct_per_line", "tk_punct_pl_sd", "ls_nopunct",
      "ls_multiclause"),
     "**标点过于规整**：逗号句号的位置像模板，每行标点数都差不多",
     "打乱标点：有的句子干脆不断句，有的用破折号/冒号/问号；别让每行都以逗号结尾",
     "**标点有个性**：位置不整齐，有的句子不断句，混用多种标点"),

    (("tl_mean", "tl_p50", "tl_p25", "tl_p75", "tl_iqr", "tl_sd", "tl_range",
      "tl_lo_frac", "tl_hi_frac", "tl_p10", "tl_p90", "tl_p95", "tl_p99",
      "tl_kurt", "tl_skew", "tl_top5", "tl_max", "tk1_frac", "tk2_frac", "tk3_frac"),
     "**字词长度分布太窄**：几乎只用两字词，没有长短混搭",
     "混入单字词（很/就/还）和四字以上的长词、专有名词",
     "**字词长短混搭**：单字词和长词、专有名词交替出现"),

    (("if_ni_decay", "if_ni_slope", "if_ni_curv", "if_ni_m", "if_ni_range",
      "if_ni_q1", "if_ni_q4", "if_head_tail_jac", "if_tail_newratio",
      "if_headtail_minus_mid", "if_burst", "if_dis"),
     "**信息推进太平顺**：从头到尾每条都在平稳叠加新信息，没有停顿和回旋",
     "允许中间有「不推进」的句子（闲笔、具体琐事、插话），别让每句都在推进主题",
     "**信息推进有停顿回旋**：有闲笔、插话、「不推进」的句子"),

    (("ls_digit", "ls_measure", "ls_material", "ls_tool", "sf_nature", "sf_social",
      "sf_timew", "ls_body", "ls_seq", "ls_causal", "ls_spatial", "ls_temporal",
      "ls_relay"),
     "**具体名词（数字／度量／材料／工具／场景）不足**，或逻辑连接词交代得太齐全",
     "加入精确到小数点的数字、具体材料/工具/地名；删掉「因此/于是/接着」这类连接词",
     "**具体名词充足**：有精确数字、材料、工具、地名；连接词不多"),

    (("tk_m", "tk_sd", "tk_kurt", "tk_skew", "tk_per_line_m", "tk_per_line_sd",
      "cz_doc", "cz_line_m", "cz_line_max", "cz_line_min", "tl_isd_m", "tl_isd_sd",
      "tl_imax_m", "tl_imax_max", "tl_ispan_m", "vo_growth_m", "vo_growth_slope",
      "vo_halfpoint", "vo_mattr_sd"),
     "**整体篇幅与节奏过于整齐**：段落规模、词汇增长曲线都很平滑",
     "让段落长短悬殊一些；词汇引入忽快忽慢，别让人一眼看出节奏",
     "**篇幅与节奏不整齐**：段落长短悬殊，词汇引入忽快忽慢"),
]

_HINTS_BY_FEAT = {}
for _names, _ai, _adv, _hu in HINTS:
    for _n in _names:
        _HINTS_BY_FEAT[_n] = (_names, _ai, _adv, _hu)


def _search_dirs():
    ds = []
    if getattr(sys, "frozen", False):
        ds.append(os.path.dirname(os.path.abspath(sys.executable)))
    ds += [HERE, os.getcwd(), os.path.join(os.path.expanduser("~"), ".jury")]
    return ds


def _find(name):
    for d in _search_dirs():
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _load_ref():
    p = _find("_ref145.pkl")
    return pickle.load(io.open(p, "rb")) if p else None


def _feats(lines):
    import jury_det as JD
    lp = _find("_det_lit_np.pkl")
    lit = JD.Detector(path=lp, model="__x__")
    L2 = [x for x in lines if str(x).strip()]
    if len(L2) < 6:
        return None, "行数太少（<6 行）"
    v = lit._raw_feats(L2)
    if v is None:
        return None, "特征抽取失败（文本过短或无中文）"
    return v, None


def analyze(lines, top=4):
    """返回 (结果, 错误)。结果含 genre / is_ai / ai_sal / hu_sal"""
    import jury_det as JD
    ref = _load_ref()
    if ref is None:
        return None, "缺少参考分布 _ref145.pkl"
    v, err = _feats(lines)
    if err:
        return None, err
    q = JD.Detector(model="auto_c").predict_lines(lines)
    tf = q.get("top_feats") or {}
    genre = tf.get("判定体裁") or "?"
    is_ai = bool(q.get("is_ai"))
    KEYS = ref["keys"]
    x = np.array([v[k] for k in KEYS], float)
    HC = ref["per_class"].get(genre + "|H") or ref["human_all"]
    AC = ref["per_class"].get(genre + "|A") or ref["ai_all"]
    mH, sH = np.array(HC["mean"]), np.array(HC["sd"])
    mA, sA = np.array(AC["mean"]), np.array(AC["sd"])
    bias = (x - mH) / sH - (x - mA) / sA

    def pick(order):
        got = []; used = set()
        for k in order:
            name = KEYS[k]
            h = _HINTS_BY_FEAT.get(name)
            if h is None:
                continue
            if any(n in used for n in h[0]):
                continue
            for n in h[0]:
                used.add(n)
            ai_dir = bias[k] > 0
            got.append({"ai": h[1], "adv": h[2], "hu": h[3],
                        "dir": "ai" if ai_dir else "human",
                        "strength": float(abs(bias[k])), "peak": name})
        return got

    ai_sal = pick(np.argsort(-bias))[:top]
    used_ai = {s["peak"] for s in ai_sal} | {n for s in ai_sal for n in _HINTS_BY_FEAT[s["peak"]][0]}
    hu_sal = []
    for s in pick(np.argsort(bias)):
        if s["peak"] in used_ai or any(n in used_ai for n in _HINTS_BY_FEAT[s["peak"]][0]):
            continue
        hu_sal.append(s)
        if len(hu_sal) >= top:
            break
    return {"genre": genre, "is_ai": is_ai, "ai_sal": ai_sal, "hu_sal": hu_sal,
            "p_class": tf.get("该类概率")}, None


def render(lines, label=None, top=4):
    r, err = analyze(lines, top=top)
    out = []
    if err:
        return "  ✗ %s" % err
    L = [x for x in lines if str(x).strip()]
    out.append("=" * 68)
    if label:
        out.append("  文本：%s（%d 行）" % (label, len(L)))
    out.append("  ── 结论 ──")
    out.append("    体裁：%s ｜ 偏向：%s" % (r["genre"], "**更像 AI**" if r["is_ai"] else "**更像人类**"))
    if r["is_ai"]:
        if r["ai_sal"]:
            out.append("")
            out.append("  ── 最像 AI 的地方（按显著程度）──")
            for i, s in enumerate(r["ai_sal"], 1):
                out.append("    %d. %s" % (i, s["ai"]))
                out.append("       → 建议：%s" % s["adv"])
        if r["hu_sal"]:
            out.append("")
            out.append("  ── 仍然像人类的地方 ──")
            for i, s in enumerate(r["hu_sal"], 1):
                out.append("    %d. %s" % (i, s["hu"]))
    else:
        if r["hu_sal"]:
            out.append("")
            out.append("  ── 最像人类的地方（按显著程度）──")
            for i, s in enumerate(r["hu_sal"], 1):
                out.append("    %d. %s" % (i, s["hu"]))
        else:
            out.append("")
            out.append("  （未发现特别像人类的突出维度 —— 属于「均衡」）")
        if r["ai_sal"]:
            out.append("")
            out.append("  ── 残余的 AI 痕迹（若想更「像人」可改进）──")
            for i, s in enumerate(r["ai_sal"], 1):
                out.append("    %d. %s" % (i, s["ai"]))
                out.append("       → 建议：%s" % s["adv"])
    out.append("")
    out.append("  （本模式只给人话诊断，不返回概率 —— 想看读数请用 `score`）")
    return "\n".join(out)
