# -*- coding: utf-8 -*-
"""§9.146 「信息流结构」特征族（纯文本，不需 GPU）
================================================================================
动机：§9.140 的「新信息率双尾异常」是唯一抓到过 HF05/HF15 的信号。
     它测的是**信息在诗中如何铺开与收束** —— 结构性、非逐行统计。
本模块把它系统化：18 个特征，覆盖
   A 词汇增长：TTR 曲线 / hapax(仅出现一次) / dis-legomena(恰好两次)
   B 新信息弧：新信息率的位置剖面（首/中/尾 + 斜率 + 曲率）
   C 复用结构：内容词在不同间隔上的复现 / 复现的突发性
   D 首尾呼应：末段与首段的重叠 / 回到开头的程度
"""
import re
import numpy as np

CJK = re.compile(r"[\u4e00-\u9fff]")
FUNC = set("的了在是我你他她它们这那有和与就不人都一也又被把从对到而之其所以为着过很"
           "会能可要还只把让给跟向于并且但或如果因为呢吗吧啊呀哦嗯着了个地得")


def _cc(s):
    """内容字列表"""
    return [c for c in s if CJK.match(c) and c not in FUNC]


def feats_infoflow(lines):
    L = [x.strip() for x in lines if x and x.strip()]
    n = len(L)
    if n < 6: return None
    CL = [_cc(x) for x in L]
    flat = [c for row in CL for c in row]
    if len(flat) < 20: return None
    N = len(flat)
    cnt = {}
    for c in flat: cnt[c] = cnt.get(c, 0) + 1
    out = {}

    # ── A 词汇增长
    uniq = len(cnt)
    out["if_ttr"] = uniq / N                                   # 型例比
    out["if_hapax"] = sum(1 for v in cnt.values() if v == 1) / uniq      # 只出现一次的字占比
    out["if_dis"] = sum(1 for v in cnt.values() if v == 2) / uniq        # 恰好两次
    out["if_rep_ratio"] = (N - uniq) / N                       # 复用率
    # TTR 曲线斜率（分 4 段）
    seg = np.array_split(np.array(flat, dtype=object), 4)
    ttr = []
    for s in seg:
        if len(s) == 0: ttr.append(0.0); continue
        ttr.append(len(set(s)) / len(s))
    ttr = np.array(ttr)
    out["if_ttr_slope"] = float(np.polyfit(np.arange(4), ttr, 1)[0])
    out["if_ttr_q1"] = float(ttr[0]); out["if_ttr_q4"] = float(ttr[-1])

    # ── B 新信息弧（逐行首次出现的比例）
    seen, ni = set(), []
    for row in CL:
        if not row: ni.append(0.0); continue
        new = [c for c in row if c not in seen]
        ni.append(len(new) / len(row)); seen.update(row)
    ni = np.array(ni)
    out["if_ni_m"] = float(ni.mean())
    out["if_ni_q1"] = float(ni[:max(1, n//4)].mean())
    out["if_ni_q4"] = float(ni[-max(1, n//4):].mean())
    out["if_ni_slope"] = float(np.polyfit(np.arange(n), ni, 1)[0]) if n > 2 else 0.0
    out["if_ni_range"] = float(ni.max() - ni.min())
    out["if_ni_decay"] = float(ni[:n//2].mean() - ni[n//2:].mean())   # 前段−后段（>0=逐渐收束）
    # 曲率：二阶差分均值
    out["if_ni_curv"] = float(np.mean(np.diff(ni, 2))) if n > 3 else 0.0

    # ── C 复用结构（内容字在不同间隔上的复现率）
    pos = {}
    for i, row in enumerate(CL):
        for c in row: pos.setdefault(c, []).append(i)
    for lag in (1, 2, 3):
        hit = 0
        for c in set(flat):
            p = pos[c]
            if len(p) >= 2:
                hit += sum(1 for a, b in zip(p, p[1:]) if b - a <= lag + 1)
        out["if_rep_lag%d" % lag] = hit / uniq if uniq else 0.0
    # 突发性：内容字出现位置的集中度（基尼）
    g = []
    for c, p in pos.items():
        g.append(len(p) / n)
    if g:
        g = np.sort(np.array(g)); m = len(g)
        out["if_burst"] = float((2 * np.sum((np.arange(1, m+1)) * g) / (m * g.sum()) - (m+1)/m)) if g.sum() > 0 else 0.0
    else:
        out["if_burst"] = 0.0

    # ── D 首尾呼应
    k = max(1, n // 4)
    head = set(c for row in CL[:k] for c in row)
    tail = set(c for row in CL[-k:] for c in row)
    out["if_head_tail_jac"] = len(head & tail) / max(1, len(head | tail))
    # 末段相对首段的「新字」比例（低=回到开头的字）
    out["if_tail_newratio"] = len(tail - head) / max(1, len(tail))
    # 全篇首尾重叠 vs 中段重叠（对照）
    mid = set(c for row in CL[k:-k] for c in row) if n > 2*k else set()
    out["if_headtail_minus_mid"] = out["if_head_tail_jac"] - (len(head & mid) / max(1, len(head | mid)))
    return out


KEYS = None
def keys():
    global KEYS
    if KEYS is None:
        r = feats_infoflow(["\u4e00\u4e8c\u4e09\u56db\u4e94\u516d", "\u4e03\u516b\u4e5d\u5341\u7532\u4e59",
                            "\u4e19\u4e01\u620a\u5df1\u5e9a\u8f9b", "\u58ec\u7678\u5b50\u4e11\u5bc5\u536f",
                            "\u8fb0\u5df3\u5348\u672a\u7533\u9149", "\u620c\u4ea5\u5b50\u4e11\u5bc5\u536f"])
        KEYS = sorted(r.keys()) if r else []
    return KEYS


# ══════════════════════════════════════════════════════════════════
# §9.152 「词汇组织 + 信息铺开」扩展族（12 维）
# ══════════════════════════════════════════════════════════════════
def feats_vocaborg(lines):
    L = [x.strip() for x in lines if x and x.strip()]
    n = len(L)
    if n < 6: return None
    CL = [_cc(x) for x in L]
    flat = [c for row in CL for c in row]
    if len(flat) < 20: return None
    N = len(flat)
    cnt = {}
    for c in flat: cnt[c] = cnt.get(c, 0) + 1
    freq = np.array(sorted(cnt.values(), reverse=True), float)
    out = {}
    # ① Zipf 斜率（频率-秩曲线）
    rank = np.arange(1, len(freq) + 1)
    try:
        out["vo_zipf"] = float(np.polyfit(np.log(rank), np.log(freq), 1)[0])
    except Exception:
        out["vo_zipf"] = 0.0
    # ② Herfindahl 集中度
    p = freq / freq.sum()
    out["vo_hhi"] = float((p ** 2).sum())
    out["vo_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
    # ③ MATTR（滑动窗口型例比，窗口 10 字）
    W = 10
    if N >= W:
        vals = [len(set(flat[i:i+W])) / W for i in range(N - W + 1)]
        out["vo_mattr_m"] = float(np.mean(vals)); out["vo_mattr_sd"] = float(np.std(vals))
        out["vo_mattr_slope"] = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
    else:
        out["vo_mattr_m"] = out["vo_mattr_sd"] = out["vo_mattr_slope"] = 0.0
    # ④ 复现距离（同一内容字两次出现之间隔多少行）
    pos = {}
    for i, row in enumerate(CL):
        for c in set(row): pos.setdefault(c, []).append(i)
    dists = [b - a for c, ps in pos.items() for a, b in zip(ps, ps[1:])]
    if dists:
        # ★ §9.177 修正：复现距离以「行」为单位、且未做长度归一化
        #   实测 corr(行数, vo_repdist_m) = **+0.898** ⇒ 伪特征（几乎纯粹编码长度）
        #   ⇒ 除以行数做归一化（改为「占全篇的比例」）
        out["vo_repdist_m"] = float(np.mean(dists)) / n
        out["vo_repdist_med"] = float(np.median(dists)) / n
        out["vo_repdist_sd"] = float(np.std(dists)) / n
        out["vo_rep_cross"] = float(np.mean([d >= 1 for d in dists]))   # 跨界复现比例
        out["vo_rep_near"] = float(np.mean([d == 1 for d in dists]))
    else:
        out["vo_repdist_m"] = out["vo_repdist_med"] = out["vo_repdist_sd"] = 0.0
        out["vo_rep_cross"] = out["vo_rep_near"] = 0.0
    # ⑤ 词汇增长速度（每行新增型数）
    seen, growth = set(), []
    for row in CL:
        new = len(set(row) - seen); growth.append(new); seen |= set(row)
    growth = np.array(growth, float)
    out["vo_growth_m"] = float(growth.mean())
    out["vo_growth_slope"] = float(np.polyfit(np.arange(n), growth, 1)[0]) if n > 2 else 0.0
    # ⑥ 半数型出现的行位置（新奇度半衰点，归一化）
    cum, half = 0, n
    tot = len(cnt)
    for i, g in enumerate(growth):
        cum += g
        if cum >= tot / 2: half = i; break
    out["vo_halfpoint"] = half / n
    # ⑦ 相邻行无共享内容字的比例
    adj = []
    for i in range(n - 1):
        a, b = set(CL[i]), set(CL[i + 1])
        adj.append(1.0 if not (a & b) else 0.0)
    out["vo_adj_disjoint"] = float(np.mean(adj)) if adj else 0.0
    return out


def keys_vocaborg():
    r = feats_vocaborg(["\u4e00\u4e8c\u4e09\u56db\u4e94\u516d", "\u4e03\u516b\u4e5d\u5341\u7532\u4e59",
                        "\u4e19\u4e01\u620a\u5df1\u5e9a\u8f9b", "\u58ec\u7678\u5b50\u4e11\u5bc5\u536f",
                        "\u8fb0\u5df3\u5348\u672a\u7533\u9149", "\u620c\u4ea5\u5b50\u4e11\u5bc5\u536f"])
    return sorted(r.keys()) if r else []


# ══════════════════════════════════════════════════════════════════
# §9.153 第三族：行内结构 + 节奏模板 + 叙事标记（22 维）
# ══════════════════════════════════════════════════════════════════
PUNCT = set("，。！？；：、…—（）《》「」『』,.!?;:")
TEMP = set("年月日号天夜早晚春夏秋冬今明昨前先后再又还已经曾将"
           "点钟分秒周星期昨今明")
SPAT = set("上下里外前后中旁内之间侧边顶底")
SEQ = ["先", "再", "然后", "接着", "后来", "最后", "于是", "又", "跟着", "之后"]
CAUS = ["因为", "所以", "由于", "因此", "于是", "导致", "使得", "为了"]
MEAS = set("个只条块片张根把台辆件双对副群堆袋瓶桶箱盘碗杯颗粒滴阵场份")
BODY = set("手脚头眼嘴脸背腰腿膝肩脖牙舌耳鼻心肝肺血骨皮肉头发臂指")
MAT = set("铁木石布纸玻璃铜铝钢泥水泥砖瓦陶瓷塑料皮棉麻丝绳索钉螺丝")


def _clauses(s):
    """按标点切分小句"""
    out, cur = [], ""
    for ch in s:
        if ch in PUNCT:
            if cur.strip(): out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip(): out.append(cur.strip())
    return out


def feats_linestruct(lines):
    L = [x.strip() for x in lines if x and x.strip()]
    n = len(L)
    if n < 6: return None
    CL = [_cc(x) for x in L]
    flat = [c for row in CL for c in row]
    if len(flat) < 20: return None
    out = {}
    # ── A 行内结构
    nc = [len(_clauses(x)) for x in L]
    out["ls_clause_m"] = float(np.mean(nc)); out["ls_clause_sd"] = float(np.std(nc))
    cl_len = [len(c) for x in L for c in _clauses(x)]
    out["ls_clauselen_m"] = float(np.mean(cl_len)) if cl_len else 0.0
    out["ls_clauselen_sd"] = float(np.std(cl_len)) if cl_len else 0.0
    out["ls_multiclause"] = float(np.mean([1.0 if c > 1 else 0.0 for c in nc]))
    # 纯标点（无标点行比例）
    out["ls_nopunct"] = float(np.mean([1.0 if not any(ch in PUNCT for ch in x) else 0.0 for x in L]))
    # ── B 节奏模板：行首/行尾字的复用
    head = [x[0] for x in L if x]; tail = [x[-1] for x in L if x]
    out["ls_head_uniq"] = len(set(head)) / max(1, len(head))      # 行首多样化
    out["ls_tail_uniq"] = len(set(tail)) / max(1, len(tail))
    out["ls_head_rep"] = 1 - out["ls_head_uniq"]
    out["ls_tail_rep"] = 1 - out["ls_tail_uniq"]
    # 行首内容字（虚词开头比例的反面）
    out["ls_head_content"] = float(np.mean([1.0 if (c and c[0] not in FUNC) else 0.0 for c in CL]))
    out["ls_tail_content"] = float(np.mean([1.0 if (c and c[-1] not in FUNC) else 0.0 for c in CL]))
    # ── C 叙事/关系标记密度（每 100 内容字）
    allt = "".join(L)
    N = max(1, len(allt))
    out["ls_temporal"] = 100.0 * sum(1 for ch in allt if ch in TEMP) / N
    out["ls_spatial"] = 100.0 * sum(1 for ch in allt if ch in SPAT) / N
    out["ls_seq"] = 100.0 * sum(allt.count(k) for k in SEQ) / N
    out["ls_causal"] = 100.0 * sum(allt.count(k) for k in CAUS) / N
    # ── D 具体名词代理
    out["ls_measure"] = 100.0 * sum(1 for ch in allt if ch in MEAS) / N
    out["ls_body"] = 100.0 * sum(1 for ch in allt if ch in BODY) / N
    out["ls_material"] = 100.0 * sum(1 for ch in allt if ch in MAT) / N
    out["ls_digit"] = 100.0 * sum(1 for ch in allt if ch.isdigit()) / N
    # ── E 行间词汇「接力」（前一行末字/首字与当前行首字/末字的关联）
    relay = 0
    for i in range(n - 1):
        a, b = CL[i], CL[i + 1]
        if a and b and (a[-1] in b or b[0] in a): relay += 1
    out["ls_relay"] = relay / max(1, n - 1)
    # ── F 行的「长度变化模式」的规律性（相邻差的符号一致性）
    ln = np.array([len(x) for x in L], float)
    d = np.diff(ln)
    if len(d) > 1:
        sign = np.sign(d); out["ls_len_alt"] = float(np.mean(sign[1:] != sign[:-1]))   # 交替程度
    else:
        out["ls_len_alt"] = 0.0
    return out


def keys_linestruct():
    r = feats_linestruct(["\u4e00\u4e8c\u4e09\u56db\u4e94\u516d", "\u4e03\u516b\u4e5d\u5341\u7532\u4e59",
                          "\u4e19\u4e01\u620a\u5df1\u5e9a\u8f9b", "\u58ec\u7678\u5b50\u4e11\u5bc5\u536f",
                          "\u8fb0\u5df3\u5348\u672a\u7533\u9149", "\u620c\u4ea5\u5b50\u4e11\u5bc5\u536f"])
    return sorted(r.keys()) if r else []


# ══════════════════════════════════════════════════════════════════
# §9.155 第五族：抽象度 / 语义场 / 标点节奏 / 虚词骨架（20 维）
# ══════════════════════════════════════════════════════════════════
ABSTRACT = set("情感情思意念义命运孤独灵魂永恒虚无存在悲伤欢喜爱恨梦想"
               "记忆时间岁月历史文明自由真理信仰希望绝望痛苦幸福")
NATURE = set("山水风雨雪云日月星辰天地草木花树石海江河湖溪雾霜露冰火")
TOOL   = set("刀斧锤钳锯钻钉螺丝扳手机床泵阀管线路灯锁钥匙袋箱盒绳网")
SOCIAL = set("人官民工农兵师医学生员警兵商客官衙门厂矿社团队组织单位")
TIME_W = set("年月日时刻分秒周季节春夏秋冬朝夕昼夜古今早晚")

def feats_semfield(lines):
    L = [x.strip() for x in lines if x and x.strip()]
    n = len(L)
    if n < 6: return None
    s = "".join(L)
    cj = [c for c in s if "\u4e00" <= c <= "\u9fff"]
    N = max(1, len(cj))
    out = {}
    # ── A 抽象度
    out["sf_abstract"] = 100.0 * sum(1 for c in cj if c in ABSTRACT) / N
    out["sf_nature"]   = 100.0 * sum(1 for c in cj if c in NATURE) / N
    out["sf_tool"]     = 100.0 * sum(1 for c in cj if c in TOOL) / N
    out["sf_social"]   = 100.0 * sum(1 for c in cj if c in SOCIAL) / N
    out["sf_timew"]    = 100.0 * sum(1 for c in cj if c in TIME_W) / N
    out["sf_conc_minus_abs"] = (out["sf_tool"] + out["sf_nature"]) - out["sf_abstract"] * 1.5
    # ── B 语义场分布熵（五类占比的熵）
    v = np.array([out["sf_abstract"], out["sf_nature"], out["sf_tool"], out["sf_social"], out["sf_timew"]], float)
    if v.sum() > 0:
        p = v / v.sum()
        out["sf_field_ent"] = float(-(p * np.log(p + 1e-12)).sum())
        out["sf_field_dom"] = float(p.max())
    else:
        out["sf_field_ent"] = 0.0; out["sf_field_dom"] = 0.0
    # ── C 标点节奏
    PUNCT1 = set("，。！？；：、…—")
    pc = [sum(1 for ch in x if ch in PUNCT1) for x in L]
    out["sf_punct_per_line_m"] = float(np.mean(pc)); out["sf_punct_per_line_sd"] = float(np.std(pc))
    allp = [ch for ch in s if ch in PUNCT1]
    out["sf_npunct"] = 100.0 * len(allp) / N
    if allp:
        import collections as _c
        cc = _c.Counter(allp); tot = len(allp)
        out["sf_comma_share"] = cc.get("，", 0) / tot
        out["sf_period_share"] = cc.get("。", 0) / tot
        p = np.array(list(cc.values()), float) / tot
        out["sf_punct_ent"] = float(-(p * np.log(p + 1e-12)).sum())
    else:
        out["sf_comma_share"] = out["sf_period_share"] = out["sf_punct_ent"] = 0.0
    out["sf_punct_uniq_per_line"] = float(np.mean([len(set(ch for ch in x if ch in PUNCT1)) for x in L]))
    # ── D 虚词骨架：抽掉内容字后骨架的长度与重复
    skel = ["".join(c for c in _cc(x)) for x in L]      # _cc 是内容字；骨架 = 原行去掉内容字
    skel = []
    for x in L:
        sk = "".join(ch for ch in x if ch not in set("，。！？；：、…—") and (ch in FUNC or not ("\u4e00" <= ch <= "\u9fff")))
        skel.append(sk)
    lens = [len(k) for k in skel]
    out["sf_skel_len_m"] = float(np.mean(lens)); out["sf_skel_len_sd"] = float(np.std(lens))
    # 骨架重复率（相同骨架的行占比）
    import collections as _c2
    cs = _c2.Counter(k for k in skel if k)
    out["sf_skel_rep"] = float(np.mean([1.0 if v > 1 else 0.0 for v in cs.values()])) if cs else 0.0
    out["sf_skel_uniq"] = len(cs) / max(1, n)
    # 骨架的「的/了/在/是」出现次数（虚词骨架的密度）
    out["sf_func_density"] = 100.0 * sum(1 for c in cj if c in FUNC) / N
    return out


def keys_semfield():
    r = feats_semfield(["\u4e00\u4e8c\u4e09\u56db\u4e94\u516d", "\u4e03\u516b\u4e5d\u5341\u7532\u4e59",
                        "\u4e19\u4e01\u620a\u5df1\u5e9a\u8f9b", "\u58ec\u7678\u5b50\u4e11\u5bc5\u536f",
                        "\u8fb0\u5df3\u5348\u672a\u7533\u9149", "\u620c\u4ea5\u5b50\u4e11\u5bc5\u536f"])
    return sorted(r.keys()) if r else []


# ══════════════════════════════════════════════════════════════════
# §9.173 「话语连接词」族（22 维）—— 跨范式通用成分的候选
# 机制：人类诗偶尔使用逻辑/话语连接词；AI 生成的诗几乎完全不用
# ══════════════════════════════════════════════════════════════════
CONN_GROUPS = {
    "caus": ["因为", "所以", "由于", "因此", "于是", "导致", "使得", "为了", "从而", "既然"],
    "adv":  ["但是", "可是", "然而", "不过", "却", "反而", "倒是", "其实"],
    "add":  ["而且", "并且", "甚至", "况且", "何况", "以及", "不仅", "不光"],
    "cond": ["如果", "只要", "除非", "要是", "假如", "倘若", "一旦"],
    "conc": ["虽然", "尽管", "即使", "哪怕", "无论", "不管", "固然"],
    "seq":  ["然后", "接着", "后来", "最后", "首先", "其次", "终于", "随即", "继而"],
    "meta": ["可以说", "应该说", "总之", "换言之", "也就是说", "值得注意", "事实上", "实际上"],
    "time": ["从前", "过去", "当年", "那时候", "如今", "现在", "后来", "起初", "当初"],
}
CN_ALL = sorted({w for v in CONN_GROUPS.values() for w in v})


def feats_discourse(lines):
    L = [x.strip() for x in lines if x and x.strip()]
    n = len(L)
    if n < 6: return None
    s = "".join(L)
    cj = [c for c in s if "\u4e00" <= c <= "\u9fff"]
    N = max(1, len(cj))
    out = {}
    alln = 0
    for g, ws in CONN_GROUPS.items():
        c = sum(s.count(w) for w in ws)
        alln += c
        out["dc_%s" % g] = 100.0 * c / N                        # 密度
        out["dc_%s_p" % g] = 1.0 if c > 0 else 0.0               # 是否出现（0/1）
    out["dc_all"] = 100.0 * alln / N
    out["dc_all_p"] = 1.0 if alln > 0 else 0.0
    out["dc_kinds"] = float(sum(1 for g in CONN_GROUPS if any(s.count(w) for w in CONN_GROUPS[g])))
    out["dc_uniq_words"] = float(sum(1 for w in CN_ALL if w in s))
    # 连接词出现在行首的比例（行首连接 = 显式逻辑衔接）
    head = sum(1 for x in L if any(x.startswith(w) for w in CN_ALL))
    out["dc_head_rate"] = head / n
    # 连接词出现在行末
    tail = sum(1 for x in L if any(x.endswith(w) for w in CN_ALL))
    out["dc_tail_rate"] = tail / n
    # 每行的连接词数（均值/离散）
    pc = np.array([sum(x.count(w) for w in CN_ALL) for x in L], float)
    out["dc_per_line_m"] = float(pc.mean()); out["dc_per_line_sd"] = float(pc.std())
    # 「显式衔接」比例：相邻行之间靠连接词衔接（而非靠意象并置）
    link = sum(1 for i in range(n - 1) if any(L[i + 1].startswith(w) for w in CN_ALL))
    out["dc_link_rate"] = link / max(1, n - 1)
    # 虚词「的/了」密度（与连接词无关的对照，但也是「书面化」指标）
    out["dc_de"] = 100.0 * s.count("的") / N
    out["dc_le"] = 100.0 * s.count("了") / N
    return out


def keys_discourse():
    r = feats_discourse(["\u4e00\u4e8c\u4e09\u56db\u4e94\u516d", "\u4e03\u516b\u4e5d\u5341\u7532\u4e59",
                         "\u4e19\u4e01\u620a\u5df1\u5e9a\u8f9b", "\u58ec\u7678\u5b50\u4e11\u5bc5\u536f",
                         "\u8fb0\u5df3\u5348\u672a\u7533\u9149", "\u620c\u4ea5\u5b50\u4e11\u5bc5\u536f"])
    return sorted(r.keys()) if r else []
