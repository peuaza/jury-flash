# -*- coding: utf-8 -*-
"""jury —— 中文 AI 文本检测器（命令行 + 交互模式 + 轻量服务）
================================================================================
两种用法
  ① 命令行（在终端里）
       jury score -f 诗.txt
       jury batch -i in.jsonl -o out.jsonl
       jury serve -p 8131
       jury info
  ② **双击 jury.exe** ⇒ 进入交互模式，直接在窗口里敲命令，可连敲多条
       jury> score -f 诗.txt
       jury> info
       jury> exit

判定口径（默认 FP 预算 5%）：p >= 阈值 ⇒ 判 AI
★ 无需 GPU、无需联网、无需安装 Python
"""
import argparse, io, json, os, shlex, sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_DET = None


_CACHE = {}


def _load(model="multi"):
    if model not in _CACHE:
        from jury_det import Detector
        _CACHE[model] = Detector(model=model)
    return _CACHE[model]


# ──────────────────────────── 子命令 ────────────────────────────
def cmd_score(a):
    d = _load(_pick_model(a))
    if a.file:
        if not os.path.exists(a.file):
            print("  ✗ 找不到文件：%s" % a.file)
            print("    （提示：路径要写全，或先把 .txt 拖到窗口里再回车）")
            return 2
        try:
            text = io.open(a.file, encoding="utf-8", errors="replace").read()
        except Exception as e:
            print("  ✗ 读取失败：%s" % e); return 2
    elif a.text:
        text = a.text.replace("\\n", "\n")
    else:
        print("  ✗ 请用 -f 文件 或 -t \"文本\"")
        return 2
    if _pick_model(a) == "auto_m":
        import jury_prep as _JP
        _t = _JP.strip_markdown(str(text)) if not getattr(a, "raw", False) else str(text)
        _L = _JP.text_to_lines(_t) if not getattr(a, "no_seg", False) else _t.splitlines()
        _D = _load("auto_m")
        _r = _D.predict_moderate(_L, a.fp)
        if not _r.get("ok"):
            print("  ✗ %s" % _r.get("error"))
            print("  （注意：auto 需要 ≥6 行、且含足够中文。想强制指定体裁可用 -m lit/expo/narr/info/classic）")
            return 2
        print("  ⇒   **%s**" % _r["verdict"])
        _pc = _r.get("p_c"); _pp = _r.get("p_p")
        print("  保守(auto_c) P=%.4f ｜ 激进(auto_p) P=%.4f ｜ %d 行 ｜ 体裁 %s" % (
            -1 if _pc is None else _pc, -1 if _pp is None else _pp,
            _r["n_lines"], (_r.get("top_feats") or {}).get("体裁")))
        if _r.get("uncertain"):
            print("  ⚠ 两个模式结论不一致 ⇒ **不确定**（建议人工复核，或换 -m auto_p 看激进判定）")
        if getattr(a, "verbose", False):
            import jury_det as _JD
            g = (_r.get("top_feats") or {}).get("体裁")
            print("  判定体裁 %s" % g)
            for m2 in ("auto_c", "auto_p"):
                o = _JD.Detector(model=m2).predict_lines(_L, a.fp)
                print("    %-7s p=%.4f ⇒ %-5s ｜ %s" % (
                    m2, o.get("p", -1), "AI" if o.get("is_ai") else "人类",
                    {k: v for k, v in (o.get("top_feats") or {}).items()}))
        return 0
    r = d.predict(text, fp_budget=a.fp, auto_segment=not getattr(a, "no_seg", False),
                  clean=not getattr(a, "raw", False))
    if not r.get("ok"):
        print("  ✗ %s" % r.get("error")); return 2
    print("  P(AI) = %.4f   ⇒   %s" % (r["p"], "**AI**" if r["is_ai"] else "人类"))
    print("  阈值 %.4f（FP 预算 %d%%）｜ %d 行" % (r["thr"], r["fp_budget"], r["n_lines"]))
    if a.verbose:
        print("  最偏离的维度：")
        for k, v in r["top_feats"].items():
            print("    %-24s z = %+.2f" % (k, v))
    return 0


def cmd_batch(a):
    d = _load(_pick_model(a))
    if not os.path.exists(a.input):
        print("  ✗ 找不到输入文件：%s" % a.input); return 2
    fo = io.open(a.output, "w", encoding="utf-8")
    n = hit = 0
    for ln in io.open(a.input, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if not ln: continue
        try: r = json.loads(ln)
        except Exception: continue
        L = r.get("lines") or [x for x in str(r.get("text", "")).splitlines() if x.strip()]
        o = d.predict_lines(L, fp_budget=a.fp, clean=not getattr(a, "raw", False))
        rec = {"key": r.get("key"), "n_lines": o.get("n_lines")}
        if o.get("ok"):
            rec.update({"p": o["p"], "is_ai": o["is_ai"], "thr": o["thr"]}); hit += int(o["is_ai"])
        else:
            rec["error"] = o.get("error")
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
        if n % 500 == 0: print("    %d 条…" % n, flush=True)
    fo.close()
    print("  完成 %d 条 ｜ 判 AI %d 条（%.0f%%）⇒ %s" % (n, hit, 100 * hit / max(1, n), a.output))
    return 0


def cmd_serve(a):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    d = _load()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": d.name})
            else:
                self._send(404, {"error": "not found"})
        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, {"ok": True, "result": d.predict(
                    req.get("text", ""), fp_budget=int(req.get("fp", a.fp)),
                    auto_segment=not req.get("no_seg", False), clean=not req.get("raw", False))})
            except Exception as e:
                self._send(500, {"error": str(e)[:200]})

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print("  jury 服务已启动 http://127.0.0.1:%d/" % a.port)
    print("    GET  /health                            健康检查")
    print('    POST /score  {"text":"...","fp":5}      判定')
    print("  （Ctrl+C 停止）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止")
    return 0


def cmd_info(a):
    """显示模型与当前设置（v1.0.0）"""
    import os, pickle
    m = getattr(a, "model", None) or _pick_model(a)
    print("  jury-flash-CN v1.0.0")
    print("  当前默认模式：%s" % m)
    print()
    # 主模型（多分类）
    try:
        import jury_det as _JD
        W = pickle.load(io.open(os.path.join(os.path.dirname(_JD.__file__), "_det_multi14.pkl"), "rb"))
        keys = W.get("keys") or []
        nsl = len([k for k in keys if str(k).startswith("sl_")])
        print("  主模型：%s" % W.get("name", "?"))
        print("  结构：%d 类（%d 体裁 × 人机）｜ %d 维特征（含 %d 维顺序 LM）" % (
            len(W.get("classes", [])), len(W.get("genres", [])), len(keys), nsl))
        print("  集成：%d 个 MLP" % len(W.get("ens", []) or [1]))
        print("  判定门槛：单类概率 ≥ %.2f ｜ AI 类总概率 ≥ %.2f" % (
            float(W.get("pthr_c", 0.7)), float(W.get("pthr_total", 0.7))))
    except Exception as e:
        print("  （主模型信息读取失败：%s）" % str(e)[:60])
    # 用户本地微调
    try:
        import jury_learn as _JL
        recs = _JL.load_learned()
        um = _JL._find("jury_model.pkl")
        if um or recs:
            print()
            print("  本地微调：已标注 %d 条 ｜ 用户模型 %s" % (len(recs), "已生效" if um else "未生成"))
    except Exception:
        pass
    print()
    print("  依赖：无 GPU ｜ 无联网 ｜ 无模型下载")
    print("  用法：jury --help  ｜  双击进入交互模式")
    return 0


def _pick_model(a):
    """命令行 -m 优先；否则用持久化的默认模型"""
    m = getattr(a, "model", None)
    if m: return m
    try:
        import jury_conf
        d = jury_conf.get_default()
        if d: return d
    except Exception:
        pass
    return "auto_m"


def cmd_setmodel(a):
    """jury.exe -m <模型>  切换默认模型"""
    import jury_conf
    m = a.set_model
    if m in ("default", "reset", "none"):
        n = jury_conf.clear_default()
        print("  已恢复内置默认（auto_m）｜ 清除 %d 个配置" % n)
        return 0
    if jury_conf.set_default(m):
        print("  ✅ 默认模型已设为 **%s**" % m)
        print("     （之后直接 `jury.exe score -f 文件` 就用它；用 -m 可临时覆盖）")
        print("     恢复内置默认：jury.exe -m default")
    else:
        print("  ✗ 无法写入配置（目录只读？）")
        return 1
    return 0


def cmd_analysis(a):
    import jury_cmds2
    return jury_cmds2.cmd_analysis(a)


def cmd_research(a):
    import jury_cmds2
    return jury_cmds2.cmd_research(a)


def cmd_learn(a):
    """★ 本地微调：把「指定体裁 + 人机标签」的样本加入训练"""
    import jury_learn
    av = []
    for f in (getattr(a, "file", None) or []):
        av += ["-f", f]
    for d in (getattr(a, "dir", None) or []):
        av += ["-d", d]
    for t in (getattr(a, "text", None) or []):
        av += ["--text", t]
    if getattr(a, "genre", None): av += ["--genre", a.genre]
    if getattr(a, "label", None): av += ["--label", a.label]
    if getattr(a, "epochs", None): av += ["--iters", str(a.epochs)]
    for k in ("list", "undo", "clear", "fit", "no_fit"):
        if getattr(a, k, False): av += ["--" + k.replace("_", "-")]
    return jury_learn.main(av)


def cmd_reset(a):
    """删除本地微调（标注 + 用户模型），恢复内置模型"""
    import jury_learn
    n = jury_learn.clear_learned()
    print("  ✓ 已清空（恢复内置模型）" if n else "  （没有可清空的）")
    return 0


def cmd_reset(a):
    import jury_ft_cli
    return jury_ft_cli.cmd_reset(a)


# ──────────────────────────── 解析器 ────────────────────────────
def build_parser():
    ap = argparse.ArgumentParser(
        prog="jury",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""jury-flash-CN v1.0.0 —— 中文 AI 文本检测器

  纯文本统计，无需 GPU、无需联网。
  原理：把文本压成 160 维可测量的统计特征，经 7 体裁 × 人机 = 14 类模型判定；
        体裁与人机同时给出（判为人类也会告知所属体裁）。""",
        epilog="""常用示例
  jury score -f 我的诗.txt                   判一篇（默认 auto_m 温和三态）
  jury score -f 我的诗.txt -m auto_c         保守模式（宁可漏检也不误报）
  jury analysis -f 我的诗.txt                人话诊断：最像 AI/人类的特征
  jury research -f 长文.txt -c -q            长文深度研究（滑窗整合 + 引用）
  jury learn -f 我的诗.txt --genre lit --label human   本地微调
  jury info                                  列出模型与当前设置

  双击本程序（或 `jury --interactive`）进入交互模式""")
    ap.add_argument("--interactive", action="store_true", help="进入交互模式")
    ap.add_argument("-V", "--version", action="version", version="jury-flash-CN v1.0.0",
                    help="显示版本号")
    ap.add_argument("-m", "--set-model", metavar="MODEL", default=None,
                    help="★ 切换默认模型（如 -m lit / -m auto_p / -m auto_c）；-m default 恢复内置默认")
    sub = ap.add_subparsers(dest="cmd", metavar="命令")

    _MODELS = ["auto_c", "auto_p", "auto_m", "auto", "lit", "expo", "narr", "info",
               "classic", "chat", "applied"]
    _MH = ("auto_m=温和三态（默认）｜auto_c=保守（宁可漏检）｜auto_p=激进｜"
           "lit/expo/narr/info/classic/chat/applied=指定体裁单头")

    s = sub.add_parser("score", help="判一篇文本",
                       description="判一篇文本。行数需 ≥6。",
                       epilog="示例  jury score -f 我的诗.txt -v\n      jury score -c   （便捷模式）")
    s.add_argument("-f", "--file", help="文件路径（.txt 等纯文本）")
    s.add_argument("-t", "--text", default="", help='直接给文本（\\n 表示换行）')
    s.add_argument("--fp", type=int, default=5, choices=[1, 2, 3, 5, 10],
                   help="允许的人类误报预算百分比（默认 5）")
    s.add_argument("-v", "--verbose", action="store_true", help="列出明细（AI 类概率 / 前三类 / 最偏离维度）")
    s.add_argument("--raw", action="store_true", help="不做预处理（不去 markdown、不自动分行）")
    s.add_argument("--no-seg", action="store_true", help="只去 markdown，不自动分行")
    s.add_argument("-c", "--convenience", action="store_true", help="★ 便捷模式：反复粘贴文件路径即测（-e 退出）")
    s.add_argument("-l", "--long", action="store_true", help="★ 长文本模式：粘贴文本，另起一行 -run 测量")
    s.add_argument("-e", "--exit", action="store_true", help="退出便捷 / 长文本模式")
    s.add_argument("-m", "--model", default=None, choices=_MODELS, help=_MH)
    s.set_defaults(fn=cmd_score)

    b = sub.add_parser("batch", help="批量判定（jsonl 输入输出）",
                       description="批量判定：输入 jsonl 每行含 lines 字段；输出含判定结果。")
    b.add_argument("-i", "--input", required=True, help="输入 jsonl")
    b.add_argument("-o", "--output", required=True, help="输出 jsonl")
    b.add_argument("--fp", type=int, default=5, choices=[1, 2, 3, 5, 10])
    b.add_argument("-m", "--model", default=None, choices=_MODELS, help=_MH)
    b.add_argument("--raw", action="store_true", help="不做预处理")
    b.set_defaults(fn=cmd_batch)

    v = sub.add_parser("serve", help="启动本地 HTTP 服务",
                       description="启动本地 HTTP 判分服务（默认 http://127.0.0.1:8131）。")
    v.add_argument("-p", "--port", type=int, default=8131, help="端口（默认 8131）")
    v.add_argument("--fp", type=int, default=5, choices=[1, 2, 3, 5, 10])
    v.set_defaults(fn=cmd_serve)

    i = sub.add_parser("info", help="显示模型与当前设置")
    i.set_defaults(fn=cmd_info)

    an = sub.add_parser("analysis", help="★ 分析模式：用人话说出文本最像 AI/人类的特征（不给概率）",
                        description="""分析模式（顾问式诊断）

  不做二分类回答，而是用**人话**列出该文本最像 AI / 最像人类的几处特征，
  并给出可操作的改进建议（参照对抗训练中的 judge 反馈）。""",
                        epilog="示例  jury analysis -f 我的诗.txt\n      jury analysis -f 长文.txt --seg 20 --top 3")
    an.add_argument("-f", "--file", nargs="*", help="文本文件（可多个；也支持目录）")
    an.add_argument("--top", type=int, default=4, help="每侧显示几条（默认 4）")
    an.add_argument("--seg", type=int, default=0, help="长文本分段宽度（0=不分段）")
    an.set_defaults(fn=cmd_analysis)

    rs2 = sub.add_parser("research", help="★ 长文深度研究：切片分析（-s 简单 / -c 全面）",
                         description="""长文深度研究（切片分析）

  -s 简单模式：按步长**切块**（不重叠），逐个给出体裁 + 人机判定。
  -c 全面模式：**滑窗**（每次下移一行）读完全文后**整合**：
              主体裁 / 副体裁 / 判 AI 的窗口占比 / 体裁段落 / 逐行整合表。""",
                         epilog="示例  jury research -f 长文.txt -s --step 20\n      jury research -f 长文.txt -c -q")
    rs2.add_argument("-f", "--file", help="文章文件（≥12 行）")
    rs2.add_argument("-s", "--simple", action="store_true", help="简单模式（切块）")
    rs2.add_argument("-c", "--comprehensive", action="store_true", help="全面模式（滑窗 + 整合）")
    rs2.add_argument("--step", type=int, default=0, help="步长 / 窗口行数（0=自动；下限 6）")
    rs2.add_argument("-q", "--quote", action="store_true", help="引用模式：显示每行开头 / 每个窗口的首尾")
    rs2.set_defaults(fn=cmd_research)

    ln = sub.add_parser("learn", help="★ 本地微调：指定样本的体裁/人机标签并加入训练",
                        description="""本地微调（把自己的样本教给检测器）

  标注「体裁 + 人类/AI」⇒ 增量重训（用随程序发布的训练特征库）
  ⇒ 产出 jury_model.pkl，**判分时自动优先加载**。

  恢复内置模型：jury reset  或  learn --clear""",
                        epilog="示例  jury learn -f 我的诗.txt --genre lit --label human\n      jury learn -d 我的诗集 --genre lit --label human\n      jury learn --list")
    ln.add_argument("-f", "--file", nargs="*", help="文件或目录（可多个）")
    ln.add_argument("-d", "--dir", nargs="*", help="目录（取其中 *.txt）")
    ln.add_argument("-t", "--text", nargs="*", help="直接给文本（\\n 分行）")
    ln.add_argument("--genre", choices=["lit", "expo", "narr", "info", "classic", "applied", "chat"],
                    help="★ 应路由到的体裁")
    ln.add_argument("--label", choices=["human", "ai"], help="★ 人类 / AI")
    ln.add_argument("--epochs", "--iters", type=int, default=1500, dest="epochs", help="重训迭代数（默认 1500）")
    ln.add_argument("--list", action="store_true", help="查看已标注样本")
    ln.add_argument("--undo", action="store_true", help="撤销最后一条")
    ln.add_argument("--clear", action="store_true", help="清空全部标注（恢复内置模型）")
    ln.add_argument("--fit", action="store_true", help="用已有标注重训（不新增）")
    ln.add_argument("--no-fit", action="store_true", help="只记录，不立即重训")
    ln.set_defaults(fn=cmd_learn)

    rs = sub.add_parser("reset", help="清除本地微调，恢复内置模型")
    rs.set_defaults(fn=cmd_reset)
    return ap


BANNER = """
  ╔══════════════════════════════════════════════════════════════════════╗
  ║   jury-flash-CN  ·  中文 AI 文本检测器                        v1.0.0  ║
  ║   纯 CPU · 无需 GPU · 无需联网 · 7 体裁 × 人机 14 类 + 160 维统计特征  ║
  ╚══════════════════════════════════════════════════════════════════════╝

  直接在下面敲命令，回车执行（可连敲多条）。输入 help 看帮助，exit 退出。

  【判分】
    score -f 文件.txt            判一篇（默认 auto_m；会告知路由到哪个体裁）
    score -f 文件.txt -v         附明细（含 AI 类概率 / 前三类）
    score -f 文件.txt -m lit     指定单个体裁头判分
    ★ score -c                   便捷模式：粘贴文件路径 + 回车即测（-e 退出）
    ★ score -l                   长文本模式：粘贴一段文本，另起一行 -run 测量
    batch -i in.jsonl -o out.jsonl   批量判定
    use <模型>                   切换本次会话默认模型
    -retry <模型>                用新模型**重测最近一次**（如 -retry auto_p）

  【分析（不给概率，只给人话）】
    ★ analysis -f 文件.txt       用人话说出该文本最像 AI / 人类的特征 + 改进建议
    ★ analysis -f 文件.txt --seg 20   长文分段分别诊断
    ★ research -f 文件.txt -s    长文切片（按步长切块，逐块给体裁 + 人机判定）
    ★ research -f 文件.txt -c    长文深度研究（滑窗 + 整合：主体裁/副体裁/概率）
                                  加 -q 可显示每行开头与每个窗口的首尾

  【本地微调（把自己的样本教给它）】
    learn -f 我的诗.txt --genre lit --label human    标注并重训
    learn --list / --undo / --clear / --fit          查看 / 撤销 / 清空 / 重训

  【其他】
    info                         列出所有模型与当前设置
    serve -p 8131                启动本地 HTTP 服务
    reset                        清除本地微调，恢复内置模型
    help                         查看完整帮助（含全部参数）
    exit                         退出
"""


def _dispatch(ap, argv):
    """执行一条命令；返回 False 表示要退出"""
    if not argv: return True
    if argv[0].lower() in ("exit", "quit", "q"): return False
    if argv[0].lower() in ("help", "?", "-h", "--help"):
        ap.print_help(); print(); return True
    try:
        a = ap.parse_args(argv)
    except SystemExit:
        return True                       # argparse 已打印错误/帮助
    if not getattr(a, "fn", None):
        ap.print_help(); return True
    try:
        a.fn(a)
    except KeyboardInterrupt:
        print("\n  （已中断）")
    except Exception as e:
        print("  ✗ 出错了：%s: %s" % (type(e).__name__, str(e)[:200]))
    return True


def interactive(ap):
    import jury_session as JS
    print(BANNER)
    sess = JS.Session(_load, _pick_model(type("X", (), {"model": None})()))
    while True:
        try:
            p = "  jury> " if sess.mode == "normal" else ("  jury[便捷]> " if sess.mode == "convenience" else "  jury[长文本]> ")
            line = input(p)
        except (EOFError, KeyboardInterrupt):
            print(); break
        try:
            if not JS.dispatch_session_line(sess, line):
                break
        except Exception as e:
            print("  ✗ 出错了：%s: %s" % (type(e).__name__, str(e)[:200]))
    print("  再见。")


def main():
    ap = build_parser()
    argv = sys.argv[1:]
    if "--interactive" in argv:
        interactive(ap); return 0
    if len(sys.argv) == 1:
        # 双击 / 无参数 ⇒ 交互模式（若是管道或非交互则打印帮助）
        try:
            tty = sys.stdin and sys.stdin.isatty()
        except Exception:
            tty = False
        if tty:
            interactive(ap)
        else:
            ap.print_help()
        return 0
    a = ap.parse_args()
    if getattr(a, "set_model", None) and not getattr(a, "cmd", None):
        return cmd_setmodel(a)
    if not getattr(a, "fn", None):
        ap.print_help(); return 0
    try:
        rc = a.fn(a)
    except KeyboardInterrupt:
        print("\n  （已中断）"); rc = 130
    except Exception as e:
        print("\n  ✗ 出错了：%s: %s" % (type(e).__name__, str(e)[:200])); rc = 1
    if sys.stdin and sys.stdin.isatty() and rc not in (0,):
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
