# -*- coding: utf-8 -*-
"""jury_det —— 纯 numpy 检测器（**不依赖 sklearn / torch / GPU / 联网**）
================================================================================
权重来自 `_det_struct_np.pkl`（由 export_weights.py 从 sklearn 模型导出，已验等价）
特征来自 structure_feats（94 维，纯文本统计）
"""
import io, os, pickle, sys
import numpy as np

def _base():
    """兼容 PyInstaller 冻结模式"""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return os.path.dirname(os.path.abspath(__file__))


_HERE = _base()
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import structure_feats as SF
import jury_prep as JP

# ── auto 模式（stacking）需要用到的子头
_AUTO_HEADS = ("poetry", "expo", "narr", "info")
# ★ §9.203 头重命名：lit（文学）/ expo（说明文）/ narr（叙事）/ info（资讯）
_AUTO_SUB = {"lit": "_det_lit_np.pkl", "expo": "_det_expo_np.pkl",
             "narr": "_det_narr_np.pkl", "info": "_det_info_np.pkl",
             "classic": "_det_classic_np.pkl", "applied": "_det_applied_np.pkl",
             "chat": "_det_chat_np.pkl"}
_ALIAS = {"flash": "lit", "poetry": "lit", "tok": "lit", "multi": "lit",
          "auto": "auto_c"}      # ★ auto 默认 = 保守版（假阳性比漏检更伤体验）

_W = os.path.join(_HERE, "_det_struct_np.pkl")
_W2 = os.path.join(_HERE, "_det_multi_np.pkl")   # §9.184 多文体 + 困难负例
_W3 = os.path.join(_HERE, "_det_rewrite_np.pkl") # §9.184 改写头
_W4 = os.path.join(_HERE, "_det_tok_np.pkl")     # §9.187 113 维（结构+token）
_W5 = os.path.join(_HERE, "_det_expo_np.pkl")    # §9.192 说明文头
_W6 = os.path.join(_HERE, "_det_flash2_np.pkl")  # §9.193 ★ flash 主头（145 维，含 CPU surprisal）
_W7 = os.path.join(_HERE, "_det_narr_np.pkl")       # §9.201 叙事头（重训版）
_W8 = os.path.join(_HERE, "_det_info_np.pkl")       # §9.195 资讯头


_MOD__ = {}


_MOD_CACHE = {}


def _find_user_flash():
    """本地微调过的 flash 模型（若存在则优先）
       查找顺序：EXE 所在目录 → 当前目录 → ~/.jury/"""
    dirs = []
    if getattr(sys, "frozen", False):
        dirs.append(os.path.dirname(os.path.abspath(sys.executable)))
    dirs += [os.getcwd(), _HERE, os.path.join(os.path.expanduser("~"), ".jury")]
    for d in dirs:
        p = os.path.join(d, "_user_flash_np.pkl")
        if os.path.exists(p): return p
    return None


def _fwd(net, X):
    """MLP 前向：relu 隐层 + logistic 输出"""
    A = X
    Ws, Bs = net["Ws"], net["Bs"]
    for i in range(len(Ws)):
        Z = A @ Ws[i] + Bs[i]
        if i < len(Ws) - 1:
            A = np.maximum(Z, 0.0) if net.get("act", "relu") == "relu" else np.tanh(Z)
        else:
            A = 1.0 / (1.0 + np.exp(-np.clip(Z, -60, 60)))
    return A[:, 0]


_USER_PROBE = [None]


def _find_user_model():
    """查找用户本地微调模型 jury_model.pkl（EXE 目录 / 工作目录 / ~/.jury）"""
    cands = []
    if getattr(sys, "frozen", False):
        cands.append(os.path.dirname(os.path.abspath(sys.executable)))
    cands += [os.getcwd(), _HERE, os.path.join(os.path.expanduser("~"), ".jury")]
    for d in cands:
        p = os.path.join(d, "jury_model.pkl")
        if os.path.exists(p):
            return p
    return None


class Detector:
    def __init__(self, path=None, fp_budget=5, model="multi"):
        """model: "multi" 多文体（AI 诗 + AI 散文）｜ "poetry" 仅诗"""
        if path is None:
            model = _ALIAS.get(model, model)
            if model == "auto_m":
                # ★ auto_m 不是单模型：它只是 auto_c + auto_p 的组合
                self.keys = []; self.thr = {"1": 0.5, "5": 0.5}
                self.name = "auto_m"; self.mean = np.array([]); self.scale = np.array([])
                self.nets = []; self.fp_budget = fp_budget
                self.use_tok = False; self.use_cpu_surp = False
                self.is_auto = True; self.W = {}; self._subs = {}
                return
            if model == "rewrite" and os.path.exists(_W3): path = _W3
            elif model in ("auto_c", "auto_p", "auto_m") and _find_user_model():
                # ★★★ 用户本地微调模型优先（jury_model.pkl）
                path = _find_user_model()
                _USER_PROBE[0] = model
            elif model == "lit" and os.path.exists(os.path.join(_HERE, "_det_lit_np.pkl")):
                path = os.path.join(_HERE, "_det_lit_np.pkl")
            elif model == "flash":
                _u = _find_user_flash()
                if _u: path = _u
                elif os.path.exists(_W6): path = _W6
                else: path = _W
            elif model in ("auto_c", "auto_p") and os.path.exists(os.path.join(_HERE, "_det_%s_np.pkl" % model)):
                path = os.path.join(_HERE, "_det_%s_np.pkl" % model)
            elif model == "auto_c" and os.path.exists(os.path.join(_HERE, "_det_auto_np.pkl")):
                path = os.path.join(_HERE, "_det_auto_np.pkl")
            elif model == "narr" and os.path.exists(_W7): path = _W7
            elif model == "info" and os.path.exists(_W8): path = _W8
            elif model in ("classic", "applied", "chat") and os.path.exists(os.path.join(_HERE, "_det_%s_np.pkl" % model)):
                path = os.path.join(_HERE, "_det_%s_np.pkl" % model)
            elif model == "flash" and os.path.exists(_W6): path = _W6
            elif model == "expo" and os.path.exists(_W5): path = _W5
            elif model == "tok" and os.path.exists(_W4): path = _W4
            elif model == "multi" and os.path.exists(_W2): path = _W2
            else: path = _W
        with io.open(path, "rb") as f:
            W = pickle.load(f)
        # ★ 用户微调模型是同一个文件 ⇒ 按请求的模式设门槛（auto_c 保守 / auto_p 激进）
        if str(path).endswith("jury_model.pkl"):
            _m = _USER_PROBE[0] or model
            W = dict(W)
            W["pthr_c"] = 0.7 if _m == "auto_c" else (0.0 if _m == "auto_p" else 0.7)
        self.is_auto = bool(W.get("kind")) and str(W.get("kind")).startswith(("router", "maxratio", "routeK", "multi14"))
        self.W = W
        self.keys = W.get("keys", [])
        self.thr = W.get("thr", {"1": 0.5, "5": 0.5})
        self.name = W.get("name", "?")
        self.mean = np.asarray(W.get("mean", []), float)
        self.scale = np.asarray(W.get("scale", []), float)
        if self.is_auto and "meta" in W:                    # router2：用元模型的尺度
            self.mean = np.asarray(W["meta"]["mean"], float)
            self.scale = np.asarray(W["meta"]["scale"], float)
        self.nets = W.get("nets", []); self.fp_budget = fp_budget
        self.use_tok = bool(W.get("use_tok", False))
        self.use_cpu_surp = bool(W.get("use_cpu_surp", False))
        self._subs = {}


    # ── auto（stacking）相关方法
    def _sub_detector(self, tag):
        if tag not in self._subs:
            p = os.path.join(_HERE, _AUTO_SUB[tag])
            self._subs[tag] = Detector(path=p, model="__load__")
        return self._subs[tag]

    def _raw_feats(self, L):
        """抽取该模型所需的原始特征字典"""
        v = SF.extract(L)
        if v is None:
            return None
        if self.use_tok:
            import toklenfeat as TL
            b = TL.feats_toklen(L)
            if b is None:
                return None
            v = dict(v); v.update(b)
        if self.use_cpu_surp:
            import toklenfeat as TL, cpu_proxy2 as CP
            CP.load_lm()
            c = CP.feats_tok_lm(L, TL._tk()); d = CP.feats_comp(L)
            if c is None or d is None:
                return None
            v.update(c); v.update(d)
            # ★ 顺序 LM 特征（跨行上下文；可选，缺失不致命）
            try:
                e = CP.feats_seq_lm(L)
                if e: v.update(e)
            except Exception:
                pass
        return v

    def _score_auto(self, L):
        """★★ 体裁路由 + 5 维 LR 元模型（router2）
            ① LR 判体裁 → ② 取该体裁头分数 → ③ LR 元模型（one-hot + 分数）
        """
        W = self.W
        if W.get("kind") == "multi14mlp":
            # ★★★★★ 14 类 MLP（反击全透明攻击：线性 gap 分解不再精确对应决策）
            KEYS = W["keys"]; G = W["genres"]; nG = W["n_genres"]
            Lit = self._sub_detector("lit")
            v = Lit._raw_feats(L)
            if v is None: return None, None
            try:
                x = np.array([[v[k] for k in KEYS]], float)
            except KeyError:
                return None, None
            if not np.isfinite(x).all(): return None, None
            z0 = (x - np.asarray(W["mean"], float)) / np.asarray(W["scale"], float)
            # ★★ 支持集成（ens: [{Ws,Bs}, ...]）⇒ 单点梯度攻击更难
            _models = W.get("ens") or [{"Ws": W["Ws"], "Bs": W["Bs"]}]
            _acc = None
            for _m in _models:
                z = z0.copy()
                Ws = _m["Ws"]; Bs = _m["Bs"]
                for k in range(len(Ws) - 1):
                    z = z @ Ws[k] + Bs[k]
                    z = np.maximum(z, 0.0)             # relu
                lo = z @ Ws[-1] + Bs[-1]
                lo = lo - lo.max()
                _p = np.exp(lo); _p = _p / _p.sum()
                _acc = _p if _acc is None else _acc + _p
            pr = (_acc / len(_models)).ravel()
            i = int(np.argmax(pr))
            genre = G[i % nG]; is_ai_cls = (i >= nG)
            _tc = float(W.get("pthr_c", 0.0))
            # ★★ 判据 B：AI 类**总概率** 也参与判定（堵住「把 AI 概率摊薄到多个类」的漏洞）
            _ta = float(W.get("pthr_total", _tc))
            _pt = float(pr[nG:].sum()) if len(pr) > nG else 0.0
            hit = bool((is_ai_cls and float(pr[i]) >= _tc) or (_pt >= _ta and _pt > 0))
            return (1.0 if hit else 0.0), {
                "genre": genre, "is_ai": hit, "hit": hit,
                "p_ai": round(float(pr[i]), 4), "pthr_c": _tc,
                "p_ai_total": round(float(pr[nG:].sum()), 4),
                "p_human_same_genre": round(float(pr[i - nG] if is_ai_cls else pr[i + nG]), 4),
                "probs": {W["classes"][k]: round(float(pr[k]), 3) for k in np.argsort(-pr)[:3]}}
        if W.get("kind") == "multi14":
            # ★★★★★ 14 类 softmax（体裁 × 人机）—— 用户要求的最终架构
            #   判定：argmax 落在 AI 类 ⇒ AI
            #   体裁：argmax 的**体裁部分** ⇒ **人类样本也有体裁**
            KEYS = W["keys"]; G = W["genres"]; nG = W["n_genres"]
            Lit = self._sub_detector("lit")
            v = Lit._raw_feats(L)
            if v is None: return None, None
            try:
                x = np.array([[v[k] for k in KEYS]], float)
            except KeyError:
                return None, None
            if not np.isfinite(x).all(): return None, None
            z = ((x - np.asarray(W["mean"], float)) / np.asarray(W["scale"], float)).ravel()
            lo = z @ np.asarray(W["coef"], float).T + np.asarray(W["intercept"], float)
            lo = lo - lo.max()
            pr = np.exp(lo); pr = (pr / pr.sum()).ravel()
            i = int(np.argmax(pr))
            genre = G[i % nG]; is_ai_cls = (i >= nG)
            # ★ 保守/激进：pthr_c（AI 类**自身概率**门槛）；0 ⇒ 纯 argmax（激进）
            _tc = float(W.get("pthr_c", 0.0))
            # ★★ 判据 B：AI 类**总概率** 也参与判定（堵住「把 AI 概率摊薄到多个类」的漏洞）
            _ta = float(W.get("pthr_total", _tc))
            _pt = float(pr[nG:].sum()) if len(pr) > nG else 0.0
            hit = bool((is_ai_cls and float(pr[i]) >= _tc) or (_pt >= _ta and _pt > 0))
            return (1.0 if hit else 0.0), {
                "genre": genre, "is_ai": hit, "hit": hit,
                "p_ai": round(float(pr[i]), 4), "pthr_c": _tc,
                "p_ai_total": round(float(pr[nG:].sum()), 4),
                "p_human_same_genre": round(float(pr[i - nG] if is_ai_cls else pr[i + nG]), 4),
                "probs": {W["classes"][k]: round(float(pr[k]), 3) for k in np.argsort(-pr)[:3]}}
        if W.get("kind") == "routeK":
            # ★★★★★ 明确路由（用户要求）：
            #   ① 总是给出体裁（判人类时也显示）⇒ route = argmax(倍数)
            #   ② 仅**被路由到的那个头**参与判定 ⇒ ratio[route] >= K[route]
            heads = W["heads"]
            # ★★★ 根治「head_thr 过期」：**始终从子头文件读阈值**，不信 W["head_thr"]
            _hthr = W.get("head_thr") or {}
            def _thr(h):
                if h in _hthr: return float(_hthr[h])
                Dh0 = self._sub_detector(h)
                return float(Dh0.thr.get("5", Dh0.thr.get(str(5), 0.5)))
            th = {h: _thr(h) for h in heads}
            Lit = self._sub_detector(heads[0])
            v = Lit._raw_feats(L)
            if v is None: return None, None
            ps = {}
            for h in heads:
                Dh = self._sub_detector(h)
                try:
                    x = np.array([[v[k] for k in Dh.keys]], float)
                except KeyError:
                    return None, None
                if not np.isfinite(x).all(): return None, None
                z = (x - Dh.mean) / Dh.scale
                ps[h] = float(np.mean([_fwd(n, z) for n in Dh.nets]))
            ratios = {h: ps[h] / max(1e-9, float(th[h])) for h in heads}
            KS = W.get("Ks") or {}
            route = max(ratios, key=lambda h: ratios[h])
            K = float(KS.get(route, W.get("K", 2.0)))
            hit = ratios[route] >= K
            p = 1.0 if hit else 0.0
            return p, {"genre": route, "head_p": round(ps[route], 5),
                       "ratio": round(ratios[route], 2), "K": K, "hit": hit,
                       "thr": float(th[route]),
                       "all_ratios": {h: round(ratios[h], 2) for h in heads},
                       "fired": ([route] if hit else [])}
        if W.get("kind") == "maxratio":
            # ★★★★★ 无训练规则：**任一**头的倍数 ≥ K ⇒ AI
            #   路由只用于「报告体裁」，不参与判定
            heads = W["heads"]; th = W["head_thr"]
            Lit = self._sub_detector(heads[0])
            v = Lit._raw_feats(L)
            if v is None: return None, None
            ps = {}
            for h in heads:
                Dh = self._sub_detector(h)
                try:
                    x = np.array([[v[k] for k in Dh.keys]], float)
                except KeyError:
                    return None, None
                if not np.isfinite(x).all(): return None, None
                z = (x - Dh.mean) / Dh.scale
                ps[h] = float(np.mean([_fwd(n, z) for n in Dh.nets]))
            ratios = {h: ps[h] / max(1e-9, float(th[h])) for h in heads}
            # ★★★ 每头独立 K（按各自人类分布标定）⇒ 加头只需算一个 K
            KS = W.get("Ks") or {}
            def _k(h):
                v = KS.get(h)
                if v is None: v = W.get("K", 2.0)
                return float(v)
            fired = [h for h in heads if ratios[h] >= _k(h)]
            best = max(fired, key=lambda h: ratios[h]) if fired else max(ratios, key=lambda h: ratios[h])
            hit = bool(fired)
            K = _k(best)
            # ★ 未触发时，报「最接近的头」但明确标注（避免把一首诗说成「说明文」）
            _near = None if fired else best
            p = 1.0 if hit else 0.0
            _g = best if hit else None
            _d = {"genre": _g, "head_p": round(ps[best], 5),
                  "ratio": round(ratios[best], 2), "K": K, "hit": hit,
                  "thr": float(th[best]), "fired": sorted(fired),
                  "all_ratios": {h: round(ratios[h], 2) for h in heads}}
            if _near: _d["near"] = _near          # 仅作参考
            return p, _d
        if W.get("kind") == "routerK":
            # ★ 无元模型：路由 + 该头倍数 ≥ K
            G = W["genre"]
            Lit = self._sub_detector(W["heads"][0])
            v = Lit._raw_feats(L)
            if v is None: return None, None
            try:
                xg = np.array([[v[k] for k in Lit.keys]], float)
            except KeyError:
                return None, None
            if not np.isfinite(xg).all(): return None, None
            # ★★★ 先算各头分数（分类器特征要用）
            _th = W["head_thr"]
            ps, xz = {}, {}
            for h in W["heads"]:
                Dh0 = self._sub_detector(h)
                try:
                    xh = np.array([[v[k] for k in Dh0.keys]], float)
                except KeyError:
                    return None, None
                if not np.isfinite(xh).all(): return None, None
                zh = (xh - Dh0.mean) / Dh0.scale
                ps[h] = float(np.mean([_fwd(n, zh) for n in Dh0.nets])); xz[h] = zh
            _ex = np.log1p(np.clip([ps[h] / max(1e-9, float(_th[h])) for h in W["heads"]], 0, 1e6))
            xg_ = np.hstack([xg[0], _ex]) if len(G["mean"]) == len(xg[0]) + len(W["heads"]) else xg[0]
            zg = (xg_.reshape(1, -1) - G["mean"]) / G["scale"]
            gi = int(np.argmax(zg @ np.asarray(G["coef"]).T + np.asarray(G["intercept"])))
            genre = W["heads"][gi]
            p_head = ps[genre]; th = float(_th[genre])
            ratio = p_head / max(1e-9, th)
            K = float(W.get("K", 1.0))
            # ★★ 判定在此完成 ⇒ 返回二值（不要再依赖 self.thr）
            hit = ratio >= K
            p = 1.0 if hit else 0.0
            return p, {"genre": genre, "head_p": round(p_head, 5),
                       "ratio": round(ratio, 2), "K": K, "hit": hit,
                       "thr": th}
        if W.get("kind") == "router2":
            Lit = self._sub_detector("lit")
            v = Lit._raw_feats(L)
            if v is None: return None, None
            try:
                xg = np.array([[v[k] for k in Lit.keys]], float)
            except KeyError:
                return None, None
            if not np.isfinite(xg).all(): return None, None
            G = W["genre"]
            zg = (xg - G["mean"]) / G["scale"]
            sc = zg @ np.asarray(G["coef"]).T + np.asarray(G["intercept"])
            gi = int(np.argmax(sc[0]))
            genre = W["heads"][gi]
            Dh = self._sub_detector(genre)
            x = np.array([[v[k] for k in Dh.keys]], float)
            if not np.isfinite(x).all(): return None, None
            z = (x - Dh.mean) / Dh.scale
            p_head = float(np.mean([_fwd(n, z) for n in Dh.nets]))
            ratio = p_head / max(1e-9, float(W["head_thr"][genre]))
            oh = np.zeros(len(W["heads"]), float); oh[gi] = 1.0
            F = np.hstack([oh, [ratio]])
            M = W["meta"]
            zm = (F - M["mean"]) / M["scale"]
            lo = float(zm @ M["coef"] + M["intercept"])
            p = 1.0 / (1.0 + np.exp(-np.clip(lo, -60, 60)))
            # ★★ 保守版必须**严格严于**激进版：加一道「倍数 ≥ 1」的闸门
            #     （否则元模型可能对某些体裁学到远小于 1 的门槛 ⇒ 反而更激进）
            if ratio < 1.0:
                p = 0.0          # ★★ 严格闸门：倍数 <1 ⇒ 直接判人类（保证 auto_c ⊆ auto_p）
            return p, {"genre": genre, "head_p": round(p_head, 5), "ratio": round(ratio, 2),
                       "thr": float(W["thr"]["5"])}
        if W.get("kind") == "router":
            Lit = self._sub_detector("lit")
            v = Lit._raw_feats(L)
            if v is None:
                return None, None
            try:
                xg = np.array([[v[k] for k in Lit.keys]], float)
            except KeyError:
                return None, None
            if not np.isfinite(xg).all():
                return None, None
            zg = (xg - W["mean"]) / W["scale"]
            scores = zg @ np.asarray(W["coef"]).T + np.asarray(W["intercept"])
            gi = int(np.argmax(scores[0]))
            genre = W["classes"][gi] if gi < len(W["classes"]) else W["heads"][gi]
            if isinstance(genre, (int, np.integer)):
                genre = W["heads"][int(genre)]
            Dh = self._sub_detector(genre)
            x = np.array([[v[k] for k in Dh.keys]], float)
            if not np.isfinite(x).all():
                return None, None
            z = (x - Dh.mean) / Dh.scale
            p = float(np.mean([_fwd(n, z) for n in Dh.nets]))
            _th = float(W["head_thr"][genre])
            return p, {"genre": genre, "head_thr": _th, "head_p": round(p, 5),
                       "ratio": round(p / max(1e-9, _th), 2)}
        # 旧版 stacking（兼容）
        ps = []
        for tag in _AUTO_HEADS:
            if tag not in _AUTO_SUB:      # ★ 旧格式子头已改名 ⇒ 直接放弃该分支
                return None, None
            D = self._sub_detector(tag)
            v = D._raw_feats(L)
            if v is None: return None, None
            x = np.array([[v[k] for k in D.keys]], float)
            if not np.isfinite(x).all(): return None, None
            z = (x - D.mean) / D.scale
            ps.append(float(np.mean([_fwd(n, z) for n in D.nets])))
        thr = [float(self.W["head_thr"].get(t, 1.0)) for t in _AUTO_HEADS]
        mul = [ps[i] / max(1e-9, thr[i]) for i in range(len(ps))]
        f = np.array([np.log1p(max(0.0, m)) for m in mul] + ps + [max(mul)], float)
        zz = (f - self.W["mean"]) / self.W["scale"]
        lo = float(zz @ self.W["coef"] + self.W["intercept"])
        return 1.0 / (1.0 + np.exp(-np.clip(lo, -60, 60))), {"raw": ps}

    def score(self, lines):
        L = [y for y in lines if y and str(y).strip()]
        if getattr(self, "is_auto", False):
            return self._score_auto(L)
        v = self._raw_feats(L)
        if v is None:
            return None, None
        if self.use_tok:
            import toklenfeat as TL
            b = TL.feats_toklen(L)
            if b is None:
                return None, None
            v = dict(v); v.update(b)
        if self.use_cpu_surp:
            import cpu_proxy2 as CP
            CP.load_lm()
            c = CP.feats_tok_lm(L, TL._tk()); d = CP.feats_comp(L)
            if c is None or d is None:
                return None, None
            v.update(c); v.update(d)
        x = np.array([[v[k] for k in self.keys]], float)
        z = (x - self.mean) / self.scale
        p = float(np.mean([_fwd(n, z) for n in self.nets]))
        return p, v

    # ── auto_m（温和）：同时跑 auto_c 与 auto_p，不一致 ⇒ 不确定
    def predict_moderate(self, lines, fp_budget=None):
        res = {}
        for tag in ("auto_c", "auto_p"):
            D = _MOD_CACHE.get(tag)
            if D is None:
                D = Detector(model=tag); _MOD_CACHE[tag] = D
            o = D.predict_lines(lines, fp_budget)
            res[tag] = o
        a, b = res["auto_c"], res["auto_p"]
        if not a.get("ok") and not b.get("ok"):
            e1 = a.get("error") or "?"; e2 = b.get("error") or "?"
            return {"ok": False,
                    "error": "auto_c: %s ｜ auto_p: %s" % (e1, e2),
                    "n_lines": a.get("n_lines", 0) or b.get("n_lines", 0)}
        ca, cb = bool(a.get("ok") and a.get("is_ai")), bool(b.get("ok") and b.get("is_ai"))
        if ca == cb:
            verdict = "AI" if ca else "人类"
        else:
            verdict = "不确定"
        return {"ok": True, "verdict": verdict, "is_ai": (verdict == "AI"),
                "uncertain": (verdict == "不确定"),
                "p_c": a.get("p"), "p_p": b.get("p"),
                "p": a.get("p") if a.get("ok") else b.get("p"),
                "thr": a.get("thr"), "fp_budget": fp_budget or self.fp_budget,
                "n_lines": a.get("n_lines") or b.get("n_lines"),
                "top_feats": {"保守(auto_c)": a.get("p"), "激进(auto_p)": b.get("p"),
                              "体裁": ((a.get("top_feats") or {}).get("判定体裁")
                                       or (a.get("top_feats") or {}).get("genre")
                                       or (b.get("top_feats") or {}).get("判定体裁"))},
                "model": "auto_m"}

    def predict_lines(self, lines, fp_budget=None, clean=True):
        q = fp_budget or self.fp_budget
        if clean:
            lines = [JP.strip_markdown(str(x)).strip() for x in lines]
        else:
            lines = [str(x) for x in lines]
        nl = len([x for x in lines if x.strip()])
        if nl < 6:
            return {"ok": False, "error": "行数太少（<6 行）", "n_lines": nl}
        p, v = self.score(lines)
        if p is None:
            return {"ok": False, "error": "特征抽取失败（文本过短或无中文）", "n_lines": nl}
        if self.is_auto and v and isinstance(v, dict) and "head_thr" in v:
            t = float(v["head_thr"])                  # ★ 路由：用该体裁头的阈值
        elif isinstance(self.thr, dict):
            t = float(self.thr.get(q, self.thr.get(str(q), 0.5)))
        else:
            t = float(self.thr)
        if self.is_auto or not self.keys:
            if isinstance(v, dict) and v.get("genre"):
                top = {"判定体裁": v.get("genre")}
                if v.get("head_p") is not None: top["该头分数"] = v.get("head_p")
                if v.get("ratio") is not None: top["相对阈值倍数"] = v.get("ratio")
                if v.get("p_ai") is not None: top["该类概率"] = v.get("p_ai")
                if v.get("p_ai_total") is not None: top["AI类概率"] = v.get("p_ai_total")
                if v.get("probs") is not None: top["前三类"] = v.get("probs")
            elif isinstance(v, dict) and v.get("near"):
                # ★ 未触发：不报体裁，只给「最接近」参考，避免把诗说成说明文
                top = {"判定体裁": "未达阈值",
                       "最接近": v.get("near"),
                       "该头倍数": v.get("ratio"),
                       "该头阈值倍数门槛": v.get("K")}
            else:
                top = {"各头分数": [round(x, 4) for x in (v.get("raw") or [])]}
        else:
            x = np.array([[v[k] for k in self.keys]], float)
            z = ((x - self.mean) / self.scale)[0]
            idx = np.argsort(-np.abs(z))[:6]
            top = {self.keys[i]: round(float(z[i]), 2) for i in idx}
        return {"ok": True, "p": round(p, 4), "is_ai": bool(p >= t), "thr": round(t, 4),
                "fp_budget": q, "n_lines": nl, "top_feats": top, "model": self.name}

    def predict(self, text, fp_budget=None, auto_segment=True, clean=True):
        """auto_segment=True ⇒ 自动按自然句分行（散文友好）；False ⇒ 只按换行
           clean=True ⇒ 先去 markdown / 富文本痕迹（默认开）
        """
        s = str(text).replace("\r\n", "\n").replace("\r", "\n")
        if clean:
            s = JP.strip_markdown(s)
        if auto_segment:
            L = JP.text_to_lines(s)
        else:
            L = s.split("\n")
        return self.predict_lines(L, fp_budget, clean=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    d = Detector()
    print("  %s ｜ %d 维 ｜ %d 子模型 ｜ 阈值 %s" % (d.name, len(d.keys), len(d.nets),
                                                 {k: round(float(v), 4) for k, v in d.thr.items()}))
    J = [json.loads(l) for l in io.open(os.path.join(_HERE, "jury_samples", "human_filtered.jsonl"),
                                        encoding="utf-8")]
    hit = sum(bool(d.predict_lines(r["lines"]).get("is_ai")) for r in J)
    print("  期末考 18 首 ⇒ 判 AI %d/18" % hit)
