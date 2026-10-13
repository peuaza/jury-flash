# -*- coding: utf-8 -*-
"""jury_ft_cli.py — 本地微调（v2：指定**体裁** + **人机标签** ⇒ 增量重训）

与旧版（§9.200c）的区别
----------------------
  旧版：只能标「是人类」，且只影响旧的 `flash2` 模型
  新版：可标 **体裁（7 类）+ 人机**，直接重训当前 multi14 集成模型

用法
----
  jury finetune -f 我的诗.txt --genre lit --label human
  jury finetune -f a.txt b.txt --genre info --label ai
  jury finetune -d 我的诗集 --genre lit --label human
  jury finetune -t "第一行\n第二行" --genre chat --label ai
  jury finetune --list           查看已标注样本
  jury finetune --undo           撤销最后一条
  jury finetune --clear          清空全部标注
  jury finetune --fit            用已有标注重训（不新增）

产物
----
  `jury_learn.jsonl`  你的标注（纯文本，可手改）
  `jury_model.pkl`    重训后的模型 ⇒ **判分时自动优先加载**
  `jury finetune --clear` + 删除 jury_model.pkl 即可完全恢复内置模型
"""
import argparse, io, json, os, pickle, sys, glob, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
if HERE not in sys.path:
    sys.path.insert(0, HERE)

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
        if os.path.exists(p): return p
    return None


# ── 标注存取 ──
def load_learned():
    p = _find(LEARN_FILE)
    if not p: return []
    out = []
    for l in io.open(p, encoding="utf-8"):
        l = l.strip()
        if l:
            try: out.append(json.loads(l))
            except Exception: pass
    return out


def append_learned(rec):
    p = learn_path()
    with io.open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def clear_learned():
    n = 0
    for f in (LEARN_FILE, USER_MODEL):
        p = _find(f)
        if p and os.path.exists(p):
            os.remove(p); n += 1
    return n


def undo_learned():
    p = _find(LEARN_FILE)
    if not p: return False
    lines = [l for l in io.open(p, encoding="utf-8").read().splitlines() if l.strip()]
    if not lines: return False
    io.open(p, "w", encoding="utf-8").write("\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""))
    return True


# ── 特征 ──
def _model_and_lit():
    import jury_det as JD
    mp = _find("_det_multi14.pkl")
    W = pickle.load(io.open(mp, "rb"))
    # 特征永远由 lit 头抽取（它带 use_tok/use_cpu_surp），与 _score_auto 一致
    lp = _find("_det_lit_np.pkl")
    lit = JD.Detector(path=lp, model="__x__")
    return W, lit


def _feats(L, W, lit):
    L2 = [str(x) for x in L if str(x).strip()]
    if len(L2) < 6: return None, "行数太少（<6 行）"
    v = lit._raw_feats(L2)
    if v is None: return None, "特征抽取失败（文本过短或无中文）"
    try:
        x = np.array([v[k] for k in W["keys"]], float)
    except KeyError:
        return None, "特征缺失"
    if not np.isfinite(x).all(): return None, "特征含非有限值"
    return x, None


# ── 重训 ──
def refit(iters=1500, verbose=True):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    bp = _find(BANK)
    if not bp:
        return False, "找不到训练库 %s（应随 jury.exe 一起发布）" % BANK
    W, lit = _model_and_lit()
    G = W["genres"]; NG = len(G); g2i = {g: i for i, g in enumerate(G)}
    d = np.load(bp)
    X = d["X"]; y = d["y"].astype(int)
    n0 = len(y)
    recs = load_learned()
    bad = 0
    for r in recs:
        g, lab = r.get("genre"), r.get("label")
        if g not in g2i or lab not in ("human", "ai"):
            bad += 1; continue
        x, err = _feats(r.get("lines") or [], W, lit)
        if x is None:
            bad += 1; continue
        cls = g2i[g] + (NG if lab == "ai" else 0)
        X = np.vstack([X, x[None, :]]); y = np.concatenate([y, [cls]])
    added = len(y) - n0
    if verbose:
        print("  已标注 %d 条（有效 %d ｜ 跳过 %d）" % (len(recs), added, bad))
        print("  训练集 %d → %d" % (n0, len(y)), flush=True)
    if added == 0:
        return False, "没有可用的新标注（请检查 --genre / --label 与文本行数）"
    t0 = time.time()
    sc = StandardScaler().fit(X); Z = sc.transform(X)
    mlps = []
    for h, s in ((96, 0), (128, 1), (64, 2)):
        m = MLPClassifier(hidden_layer_sizes=(h,), max_iter=iters, random_state=s).fit(Z, y)
        mlps.append(m)
        if verbose: print("    MLP(%d) 完成（%.0fs）" % (h, time.time() - t0), flush=True)
    out = dict(W)
    out["kind"] = "multi14mlp"
    out["mean"] = np.array(sc.mean_, float); out["scale"] = np.array(sc.scale_, float)
    out["Ws"] = [np.array(z, float) for z in mlps[0].coefs_]
    out["Bs"] = [np.array(z, float) for z in mlps[0].intercepts_]
    out["ens"] = [{"Ws": [np.array(z, float) for z in m.coefs_],
                   "Bs": [np.array(z, float) for z in m.intercepts_]} for m in mlps]
    out["name"] = "%s+learn(%d)" % (W.get("name", "?"), added)
    out["n_learned"] = added
    op = user_model_path()
    pickle.dump(out, io.open(op, "wb"), protocol=4)
    if verbose: print("  ⇒ 已保存 %s（%.0fs）" % (op, time.time() - t0))
    return True, op


# ── 收集文本 ──
def collect_texts(a):
    res = []
    for f in (a.file or []):
        if os.path.isdir(f):
            for g in sorted(glob.glob(os.path.join(f, "*.txt"))):
                res.append((g, [x.strip() for x in io.open(g, encoding="utf-8", errors="replace").read().split("\n") if x.strip()]))
        elif os.path.exists(f):
            res.append((f, [x.strip() for x in io.open(f, encoding="utf-8", errors="replace").read().split("\n") if x.strip()]))
        else:
            print("  ⚠ 跳过（找不到）：%s" % f)
    for d in (a.dir or []):
        for g in sorted(glob.glob(os.path.join(d, "*.txt"))):
            res.append((g, [x.strip() for x in io.open(g, encoding="utf-8", errors="replace").read().split("\n") if x.strip()]))
    for t in (a.text or []):
        res.append(("<text>", [x.strip() for x in str(t).replace("\\n", "\n").split("\n") if x.strip()]))
    return res


# ── 命令 ──
def cmd_finetune(a):
    if getattr(a, "list", False):
        recs = load_learned()
        um = _find(USER_MODEL)
        print("  ╔══ 本地标注 %d 条 ═══" % len(recs))
        for i, r in enumerate(recs, 1):
            L = r.get("lines") or []
            print("  ║ %2d. 体裁 %-8s ｜ %-5s ｜ %2d 行 ｜ %s" % (
                i, r.get("genre"), r.get("label"), len(L), (L[0][:36] if L else "")))
        print("  ╚" + "═" * 46)
        print("  标注文件：%s" % learn_path())
        print("  用户模型：%s" % (um if um else "（尚未生成）"))
        return 0
    if getattr(a, "clear", False):
        n = clear_learned()
        print("  ✓ 已清空 %d 个文件（恢复内置模型）" % n if n else "  （没有可清空的）")
        return 0
    if getattr(a, "undo", False):
        print("  ✓ 已撤销最后一条" if undo_learned() else "  （没有可撤销的）")
        return 0
    if getattr(a, "fit", False):
        ok, msg = refit(iters=getattr(a, "epochs", 1500))
        print(("  ✓ " if ok else "  ✗ ") + str(msg)); return 0 if ok else 2

    # 需要体裁与标签
    if not getattr(a, "genre", None) or not getattr(a, "label", None):
        print("  ✗ 请指定 --genre 与 --label")
        print("     体裁：%s" % " / ".join(GENRES))
        print("     标签：human / ai")
        print("     例：jury finetune -f 我的诗.txt --genre lit --label human")
        return 2
    items = collect_texts(a)
    if not items:
        print("  ✗ 没有可用的文本（用 -f / -d / -t 提供）"); return 2
    W, lit = _model_and_lit()
    okn = 0
    for name, L in items:
        x, err = _feats(L, W, lit)
        if x is None:
            print("  ⚠ %s：%s" % (os.path.basename(name), err)); continue
        append_learned({"genre": a.genre, "label": a.label, "n_lines": len(L),
                        "lines": L, "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
        okn += 1
    print("  ✓ 已记录 %d 条（体裁 %s ｜ %s）" % (okn, a.genre, a.label))
    print("    标注文件：%s" % learn_path())
    if okn == 0: return 2
    if getattr(a, "no_fit", False):
        print("    （未重训；稍后 `jury finetune --fit` 生效）"); return 0
    print("  正在重训（约 1~4 分钟）…", flush=True)
    ok, msg = refit(iters=getattr(a, "epochs", 1500))
    print(("  ✓ " if ok else "  ✗ ") + str(msg))
    if ok: print("  ★ 之后判分会**自动优先加载** %s" % USER_MODEL)
    return 0 if ok else 2


def cmd_reset(a):
    n = clear_learned()
    print("  ✓ 已删除本地微调（标注 + 用户模型），恢复内置模型" if n else "  （没有需要删除的）")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(prog="jury finetune")
    ap.add_argument("-f", "--file", nargs="*", help="文件或目录")
    ap.add_argument("-d", "--dir", nargs="*", help="目录（取其中 *.txt）")
    ap.add_argument("-t", "--text", nargs="*", help="直接给文本（\\n 分行）")
    ap.add_argument("--genre", choices=GENRES, help="★ 应路由到的体裁")
    ap.add_argument("--label", choices=["human", "ai"], help="★ 人类 / AI")
    ap.add_argument("--epochs", type=int, default=1500, help="重训迭代数")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--no-fit", action="store_true", help="只记录，不立即重训")
    a = ap.parse_args(argv)
    return cmd_finetune(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
