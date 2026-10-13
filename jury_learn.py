# -*- coding: utf-8 -*-
"""jury_learn.py — 本地微调：指定样本的「体裁 + 人机标签」⇒ 增量重训

用法
----
  jury learn -f 我的诗.txt --genre lit --label human
  jury learn -f a.txt b.txt --genre info --label ai
  jury learn -d 我的目录 --genre lit --label human
  jury learn -t "第一行\n第二行" --genre chat --label ai
  jury learn --list / --undo / --clear / --fit [--no-fit]

产物
----
  jury_learn.jsonl  你的标注（纯文本，可手改）
  jury_model.pkl    重训后的模型 ⇒ 判分时**自动优先加载**
  删除这两个文件即完全恢复内置模型（或 `jury reset`）
"""
import io, json, os, sys, glob, pickle, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GENRES = ["lit", "expo", "narr", "info", "classic", "applied", "chat"]
LEARN_FILE = "jury_learn.jsonl"
USER_MODEL = "jury_model.pkl"
BANK = "_train_bank.npz"


# ── 路径 ──
def _exe_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return HERE


def _writable_dir():
    d = _exe_dir()
    try:
        t = os.path.join(d, ".w_test")
        open(t, "w").close(); os.remove(t)
        return d
    except Exception:
        d2 = os.path.join(os.path.expanduser("~"), ".jury")
        os.makedirs(d2, exist_ok=True)
        return d2


def learn_path():
    return os.path.join(_writable_dir(), LEARN_FILE)


def user_model_path():
    return os.path.join(_writable_dir(), USER_MODEL)


def _find(name):
    for d in (_exe_dir(), HERE, _writable_dir(), os.getcwd()):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


# ── 标注存取 ──
def load_learned(base=None):
    p = _find(LEARN_FILE)
    if not p:
        return []
    try:
        txt = io.open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return []
    if chr(0) in txt[:400]:          # 二进制（异常情况）
        return []
    out = []
    for l in txt.splitlines():
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
            if isinstance(r, dict) and r.get("lines"):
                out.append(r)
        except Exception:
            pass
    return out


def append_learned(rec):
    p = learn_path()
    with io.open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def clear_learned():
    n = 0
    for name in (LEARN_FILE, USER_MODEL):
        p = _find(name)
        if p and os.path.exists(p):
            try:
                os.remove(p); n += 1
            except Exception:
                pass
    return n


def undo_learned(base=None):
    p = _find(LEARN_FILE)
    if not p:
        return False
    lines = [l for l in io.open(p, encoding="utf-8").read().splitlines() if l.strip()]
    if not lines:
        return False
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""))
    return True


# ── 特征抽取（与线上 _score_auto 同源：特征由 lit 子头抽）──
_LIT = None


def _lit():
    global _LIT
    if _LIT is None:
        import jury_det as JD
        lp = _find("_det_lit_np.pkl") or os.path.join(HERE, "_det_lit_np.pkl")
        _LIT = JD.Detector(path=lp, model="__x__")
    return _LIT


def _model_W():
    # 优先 multi14；若不在包内，退回 auto_c/auto_p（二者 keys/classes/genres 相同）
    for name in ("_det_multi14.pkl", "_det_auto_c_np.pkl", "_det_auto_p_np.pkl"):
        p = _find(name)
        if p:
            W = pickle.load(io.open(p, "rb"))
            if "keys" in W and "genres" in W:
                return W
    raise FileNotFoundError("找不到模型文件（_det_multi14.pkl / _det_auto_c_np.pkl）")


def _feats(L, W=None):
    """返回 (145 维向量, None) 或 (None, 错误信息)"""
    if W is None:
        W = _model_W()
    L2 = [str(x) for x in L if str(x).strip()]
    if len(L2) < 6:
        return None, "行数太少（<6 行，当前 %d 行）" % len(L2)
    v = _lit()._raw_feats(L2)
    if v is None:
        return None, "特征抽取失败（文本过短或无中文）"
    try:
        x = np.array([v[k] for k in W["keys"]], float)
    except KeyError as e:
        return None, "特征缺失：%s" % e
    if not np.isfinite(x).all():
        return None, "特征含非有限值"
    return x, None



# ── 纯 numpy 的 MLP 训练（EXE 内无 sklearn 时使用）──
def _train_mlp_np(X, y, hidden, seed, epochs=400, lr=2e-3, l2=1e-5):
    """ReLU + softmax，Adam；返回 (Ws, Bs)"""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    nc = int(y.max()) + 1
    # 类别权重（平衡）
    cnt = np.bincount(y, minlength=nc).astype(float)
    w_cls = (n / (nc * np.maximum(cnt, 1.0)))
    sw = w_cls[y]
    # He 初始化
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden)); B1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, nc)); B2 = np.zeros(nc)
    ps = [W1, B1, W2, B2]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    Y = np.zeros((n, nc)); Y[np.arange(n), y] = 1.0
    bs = 512
    step = 0
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            xb = X[j]; yb = y[j]; swb = sw[j]; yoh = Y[j]
            h_pre = xb @ W1 + B1
            h = np.maximum(h_pre, 0.0)
            lo = h @ W2 + B2
            lo -= lo.max(1, keepdims=True)
            e = np.exp(lo); pr = e / e.sum(1, keepdims=True)
            # 加权交叉熵梯度
            g = (pr - yoh) * swb[:, None] / swb.sum()
            gW2 = h.T @ g + l2 * W2; gB2 = g.sum(0)
            gh = (g @ W2.T) * (h_pre > 0)
            gW1 = xb.T @ gh + l2 * W1; gB1 = gh.sum(0)
            step += 1
            for k, gr in enumerate([gW1, gB1, gW2, gB2]):
                ms[k] = b1 * ms[k] + (1 - b1) * gr
                vs[k] = b2 * vs[k] + (1 - b2) * (gr * gr)
                mh = ms[k] / (1 - b1 ** step); vh = vs[k] / (1 - b2 ** step)
                ps[k] -= lr * mh / (np.sqrt(vh) + eps)
        W1, B1, W2, B2 = ps
    return ps

# ── 重训 ──
def refit(base=None, iters=1500, verbose=True):
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPClassifier
        _HAVE_SK = True
    except Exception:
        _HAVE_SK = False
    bp = _find(BANK)
    if not bp:
        return False, "找不到训练库 %s（应随 jury.exe 一起发布）" % BANK
    W = _model_W()
    G = W["genres"]; NG = len(G); g2i = {g: i for i, g in enumerate(G)}
    d = np.load(bp)
    X = d["X"]; y = d["y"].astype(int)
    n0 = len(y)
    recs = load_learned()
    added = 0; bad = 0
    for r in recs:
        g, lab = r.get("genre"), r.get("label")
        if g not in g2i or lab not in ("human", "ai"):
            bad += 1; continue
        x, err = _feats(r.get("lines") or [], W)
        if x is None:
            bad += 1; continue
        cls = g2i[g] + (NG if lab == "ai" else 0)
        X = np.vstack([X, x[None, :]])
        y = np.concatenate([y, [cls]])
        added += 1
    if verbose:
        print("  标注 %d 条（有效 %d ｜ 跳过 %d）" % (len(recs), added, bad))
        print("  训练集 %d → %d" % (n0, len(y)), flush=True)
    if added == 0:
        return False, "没有可用的新增标注（检查 --genre/--label，或文本是否 ≥6 行）"
    t0 = time.time()
    if _HAVE_SK:
        sc = StandardScaler().fit(X); Z = sc.transform(X)
        mean_, scale_ = np.array(sc.mean_, float), np.array(sc.scale_, float)
        models = []
        for h, s in ((96, 0), (128, 1), (64, 2)):
            m = MLPClassifier(hidden_layer_sizes=(h,), max_iter=iters, random_state=s).fit(Z, y)
            models.append({"Ws": [np.array(z, float) for z in m.coefs_],
                           "Bs": [np.array(z, float) for z in m.intercepts_]})
            if verbose:
                print("    MLP(%d) 完成（%.0fs，sklearn）" % (h, time.time() - t0), flush=True)
    else:
        mean_ = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
        scale_ = sd
        Z = (X - mean_) / scale_
        models = []
        for h, s in ((96, 0), (128, 1), (64, 2)):
            W1, B1, W2, B2 = _train_mlp_np(Z, y, h, s, epochs=min(max(iters, 200), 600))
            models.append({"Ws": [np.array(W1, float), np.array(W2, float)],
                           "Bs": [np.array(B1, float), np.array(B2, float)]})
            if verbose:
                print("    MLP(%d) 完成（%.0fs，numpy）" % (h, time.time() - t0), flush=True)
    out = dict(W)
    out["kind"] = "multi14mlp"
    out["mean"] = mean_
    out["scale"] = scale_
    out["Ws"] = models[0]["Ws"]
    out["Bs"] = models[0]["Bs"]
    out["ens"] = models
    out["name"] = "%s+learn(%d)" % (W.get("name", "?"), added)
    out["n_learned"] = added
    op = user_model_path()
    try:
        pickle.dump(out, io.open(op, "wb"), protocol=4)
    except Exception:
        op = os.path.join(HERE, USER_MODEL)
        pickle.dump(out, io.open(op, "wb"), protocol=4)
    if verbose:
        print("  => 已保存 %s（%.0fs）" % (op, time.time() - t0))
    return True, op


# ── CLI ──
def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="jury learn", add_help=True)
    ap.add_argument("-f", "--file", nargs="*", help="文件或目录")
    ap.add_argument("-d", "--dir", nargs="*", help="目录（取其中 *.txt）")
    ap.add_argument("--text", nargs="*", help="直接给文本（\\n 分行）")
    ap.add_argument("--genre", choices=GENRES, help="★ 应路由到的体裁")
    ap.add_argument("--label", choices=["human", "ai"], help="★ 人类 / AI")
    ap.add_argument("--iters", type=int, default=1500, help="重训迭代数")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--no-fit", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        recs = load_learned()
        um = _find(USER_MODEL)
        if not recs:
            print("  还没有任何本地标注")
        else:
            print("  ===== 本地标注 %d 条 =====" % len(recs))
            for i, r in enumerate(recs, 1):
                L = r.get("lines") or []
                print("   %2d. 体裁 %-8s | %-5s | %2d 行 | %s" % (
                    i, r.get("genre"), r.get("label"), len(L), (L[0][:36] if L else "")))
        print("  标注文件：%s" % learn_path())
        print("  用户模型：%s" % (um if um else "（尚未生成）"))
        return 0
    if a.clear:
        n = clear_learned()
        print("  已清空 %d 个文件（恢复内置模型）" % n if n else "  没有可清空的")
        return 0
    if a.undo:
        print("  已撤销最后一条" if undo_learned() else "  没有可撤销的")
        return 0
    if a.fit:
        ok, msg = refit(iters=a.iters)
        print(("  OK " if ok else "  FAIL ") + str(msg))
        return 0 if ok else 2

    if not a.genre or not a.label:
        print("  请指定 --genre 与 --label")
        print("     体裁：%s" % " / ".join(GENRES))
        print("     标签：human / ai")
        print("     例：jury learn -f 我的诗.txt --genre lit --label human")
        return 2

    items = []
    for f in (a.file or []):
        if os.path.isdir(f):
            for g in sorted(glob.glob(os.path.join(f, "*.txt"))):
                items.append((g, _rd(g)))
        elif os.path.exists(f):
            items.append((f, _rd(f)))
        else:
            print("  跳过（找不到）：%s" % f)
    for d in (a.dir or []):
        for g in sorted(glob.glob(os.path.join(d, "*.txt"))):
            items.append((g, _rd(g)))
    for t in (a.text or []):
        items.append(("<text>", [x.strip() for x in str(t).replace("\\n", "\n").split("\n") if x.strip()]))
    if not items:
        print("  没有可用文本（用 -f / -d / --text 提供）")
        return 2

    W = _model_W()
    okn = 0
    for name, L in items:
        x, err = _feats(L, W)
        if x is None:
            print("  跳过 %s：%s" % (os.path.basename(name), err))
            continue
        append_learned({"genre": a.genre, "label": a.label, "n_lines": len(L),
                        "lines": L, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        okn += 1
    print("  已记录 %d 条（体裁 %s | %s）" % (okn, a.genre, a.label))
    print("  标注文件：%s" % learn_path())
    if okn == 0:
        return 2
    if a.no_fit:
        print("  未重训；稍后 `jury learn --fit` 生效")
        return 0
    print("  正在重训（约 1~4 分钟）...", flush=True)
    ok, msg = refit(iters=a.iters)
    print(("  OK " if ok else "  FAIL ") + str(msg))
    if ok:
        print("  ★ 之后判分会自动优先加载 %s" % USER_MODEL)
    return 0 if ok else 2


def _rd(p):
    return [x.strip() for x in io.open(p, encoding="utf-8", errors="replace").read().split("\n") if x.strip()]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
