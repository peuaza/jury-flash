# -*- coding: utf-8 -*-
"""jury-flash-CN v1.0.0 交互模式
   ★ score -c   便捷模式：直接粘贴文件路径 ⇒ 回车即测（选项加在路径后）
   ★ score -l   长文本模式：粘贴多行文本，另起一行输入 -run 才测量
   ★ score -e   退出以上模式（回到普通交互）
   ★ -retry MODEL   用新模型重测**最近一次**输入
   ★ auto 模式明确告知**路由到哪个体裁头**
"""
import io, json, os, re, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


class Session:
    """交互会话状态"""

    def __init__(self, load_fn, model_default):
        self.load = load_fn                 # (model) -> Detector
        self.model = model_default
        self.mode = "normal"                # normal | convenience | longtext
        self.buf = []                       # 长文本缓冲
        self.last = None                    # 最近一次测量：dict(kind, value, model, opts)
        self.last_out = None

    # ── 通用：跑一次并记录
    def run_text(self, text, model=None, opts=None):
        opts = opts or {}
        model = model or self.model
        import jury_det, jury_prep as JP
        D = self.load(model)
        if model == "auto_m":
            t = JP.strip_markdown(str(text)) if not opts.get("raw") else str(text)
            L = JP.text_to_lines(t) if not opts.get("no_seg") else t.splitlines()
            r = D.predict_moderate(L, opts.get("fp", 5))
            if not r.get("ok"):
                print("  ✗ %s" % r.get("error")); return r
            print("  ⇒   **%s**" % r["verdict"])
            g = (r.get("top_feats") or {}).get("体裁")
            print("  路由到 **%s** 头 ｜ 保守 P=%.4f ｜ 激进 P=%.4f ｜ %d 行" % (
                g, r["p_c"] or -1, r["p_p"] or -1, r["n_lines"]))
            if r.get("uncertain"):
                print("  ⚠ 保守/激进不一致 ⇒ 不确定（-retry auto_p 看激进判定）")
            self.last = {"kind": "text", "value": text, "model": model, "opts": opts}
            self.last_out = r
            return r
        r = D.predict(text, fp_budget=opts.get("fp", 5),
                      auto_segment=not opts.get("no_seg", False),
                      clean=not opts.get("raw", False))
        if not r.get("ok"):
            print("  ✗ %s" % r.get("error")); return r
        print("  P(AI) = %.4f   ⇒   %s" % (r["p"], "**AI**" if r["is_ai"] else "人类"))
        tf = r.get("top_feats") or {}
        g = tf.get("判定体裁") or tf.get("genre")
        if g:
            print("  路由到 **%s** 头 ｜ 该头 P=%.4f ｜ 倍数 %.1f ｜ %d 行" % (
                g, tf.get("该头分数") or 0, tf.get("相对阈值倍数") or 0, r["n_lines"]))
        else:
            print("  阈值 %.4f（FP %d%%）｜ %d 行" % (r["thr"], r["fp_budget"], r["n_lines"]))
        if opts.get("verbose"):
            for k, v in tf.items():
                print("    %-22s %s" % (k, v))
        self.last = {"kind": "text", "value": text, "model": model, "opts": opts}
        self.last_out = r
        return r

    def run_file(self, path, model=None, opts=None):
        if not os.path.exists(path):
            print("  ✗ 找不到文件：%s" % path)
            return None
        try:
            text = io.open(path, encoding="utf-8", errors="replace").read()
        except Exception as e:
            print("  ✗ 读取失败：%s" % e); return None
        print("  ── %s ──" % os.path.basename(path))
        r = self.run_text(text, model, opts)
        if self.last: self.last["kind"] = "file"; self.last["value"] = text
        return r

    # ── 便捷模式：一行 = 一个路径（可选跟选项）
    def convenience_line(self, line):
        try:
            parts = shlex.split(line, posix=False)
        except Exception:
            parts = line.split()
        parts = [p.strip('"').strip("'") for p in parts]
        if not parts: return True
        path = parts[0]
        opts = self._parse_opts(parts[1:])
        self.run_file(path, opts=opts)
        return True

    # ── 长文本模式：累积多行，遇 -run 才测量
    def longtext_line(self, line):
        s = line.strip()
        # ★ 以 - 开头的短命令先处理（否则会被误当缓冲内容）
        if s.startswith("-retry"):
            rest = s[6:].strip().split()
            if rest: self.retry(rest[0], rest[1:])
            else: print("  ✗ 用法：-retry <模型>")
            return True
        if s.startswith("-run"):
            rest = s[4:].strip()
            try:
                more = shlex.split(rest, posix=False) if rest else []
            except Exception:
                more = rest.split()
            more = [p.strip('"').strip("'") for p in more]
            opts = self._parse_opts(more)
            text = "\n".join(self.buf)
            if not text.strip():
                print("  ✗ 缓冲区为空（先粘贴文本）")
            else:
                print("  ── 长文本（%d 行）──" % len(self.buf))
                self.run_text(text, opts=opts)
            self.buf = []
            return True
        if s in ("-e", "-exit", "-cancel"):
            self.buf = []; self.mode = "normal"
            print("  已退出长文本模式（缓冲已清空）")
            return True
        try:
            line = line.encode("utf-8", "ignore").decode("utf-8", "ignore")
        except Exception:
            line = ""
        self.buf.append(line)
        if len(self.buf) % 5 == 0:
            print("    …已累积 %d 行（输入 -run 测量，-e 取消）" % len(self.buf))
        return True

    def _parse_opts(self, toks):
        o = {}
        for t in toks:
            t = t.strip()
            if t in ("-m", "--model") : o["_need_model"] = True; continue
            if t.startswith("-m="): o["model"] = t[3:]; continue
            if t in ("--raw",): o["raw"] = True
            elif t in ("--no-seg", "--noseg"): o["no_seg"] = True
            elif t in ("-v", "--verbose"): o["verbose"] = True
            elif t.startswith("--fp="): o["fp"] = int(t[5:])
            elif t.startswith("--fp"): pass
        # -m MODEL 两段式
        for i, t in enumerate(toks):
            if t in ("-m", "--model") and i + 1 < len(toks):
                o["model"] = toks[i + 1]
        o.pop("_need_model", None)
        return o

    # ── -retry MODEL
    def retry(self, model, extra=None):
        if not self.last:
            print("  ✗ 还没有可重测的记录"); return
        opts = dict(self.last.get("opts") or {})
        if extra: opts.update(self._parse_opts(extra))
        print("  ⟳ 用 **%s** 重测最近一次输入…" % model)
        self.run_text(self.last["value"], model=model, opts=opts)


def dispatch_session_line(sess, line):
    """交互模式一行；返回 False 表示退出程序"""
    s = line.strip()
    if not s: return True
    low = s.lower()

    # 模式内
    if sess.mode == "longtext":
        return sess.longtext_line(line)
    if sess.mode == "convenience":
        if low in ("-e", "-exit", "exit", "q", "quit"):
            sess.mode = "normal"; print("  已退出便捷模式"); return True
        if low.startswith("-retry"):
            rest = s[6:].strip().split()
            if rest: sess.retry(rest[0], rest[1:])
            else: print("  ✗ 用法：-retry <模型>")
            return True
        return sess.convenience_line(line)

    # 普通模式
    if low in ("exit", "quit", "q"): return False
    if low in ("help", "?", "-h", "--help"):
        print(HELP); return True
    if low.startswith("-retry"):
        rest = s[6:].strip().split()
        if rest: sess.retry(rest[0], rest[1:])
        else: print("  ✗ 用法：-retry <模型>（如 -retry auto_p）")
        return True
    if low.startswith("-m ") and len(s.split()) == 2:
        # 允许「-m lit」直接切换本次默认
        sess.model = s.split()[1]; print("  已切换默认模型 ⇒ %s" % sess.model); return True

    try:
        toks = shlex.split(s, posix=False)
    except Exception:
        toks = s.split()
    toks = [t.strip('"').strip("'") for t in toks]
    if not toks: return True
    cmd = toks[0].lower()

    # score -c / -l / -e
    if cmd == "score" and len(toks) >= 2 and toks[1] in ("-c", "-l", "-e"):
        sub = toks[1]
        if sub == "-c":
            sess.mode = "convenience"
            print("  ★ 便捷模式：把文件路径粘贴进来、按回车即测")
            print("     想加选项就写在路径后面（--raw / --no-seg / -v / -m lit）")
            print("     退出：-e")
            return True
        if sub == "-l":
            sess.mode = "longtext"; sess.buf = []
            print("  ★ 长文本模式：粘贴一段文本，然后**另起一行输入 -run** 开始测量")
            print("     选项写在 -run 后面（如 -run --raw -m lit）；-e 取消")
            return True
        if sub == "-e":
            sess.mode = "normal"; sess.buf = []
            print("  已退出当前模式"); return True

    # 普通 score
    if cmd == "score":
        rest = toks[1:]
        opts = sess._parse_opts(rest)
        model = opts.get("model") or sess.model
        if "-f" in rest or "--file" in rest:
            i = rest.index("-f") if "-f" in rest else rest.index("--file")
            if i + 1 < len(rest):
                sess.run_file(rest[i + 1], model=model, opts=opts); return True
        if "-t" in rest or "--text" in rest:
            i = rest.index("-t") if "-t" in rest else rest.index("--text")
            if i + 1 < len(rest):
                sess.run_text(rest[i + 1].replace("\\n", "\n"), model=model, opts=opts); return True
        # 裸路径也算
        if rest and os.path.exists(rest[0]):
            sess.run_file(rest[0], model=model, opts=opts); return True
        print("  ✗ 用法：score -f 文件 ｜ score -t \"文本\" ｜ score -c ｜ score -l")
        return True

    if cmd in ("info", "model", "models"):
        import jury_det
        for m in ("auto_m", "auto_c", "auto_p", "lit", "expo", "narr", "info", "classic", "chat", "applied"):
            try:
                D = jury_det.Detector(model=m)
                if isinstance(D.thr, dict) and "5" in D.thr and len(D.thr) > 1:
                    t = "%.4f" % float(D.thr["5"])
                elif getattr(D, "W", None) and D.W.get("head_thr"):
                    t = "按头（%s）" % " ｜ ".join("%s %.3g" % (k, v) for k, v in D.W["head_thr"].items())
                else:
                    t = str(D.thr)
                print("    %-8s %-14s %s" % (m, D.name, t))
            except Exception:
                pass
        print("  当前默认：%s" % sess.model)
        return True

    if cmd == "use":
        if len(toks) >= 2:
            sess.model = toks[1]; print("  已切换默认模型 ⇒ %s" % sess.model)
        else:
            print("  当前默认：%s" % sess.model)
        return True

    # 裸路径（便捷）
    if os.path.exists(toks[0]):
        sess.convenience_line(line); return True

    print("  ✗ 未知命令：%s（输入 help）" % cmd)
    return True


HELP = """
  ── 判分 ──
    score -f 文件.txt          判一个文件（默认 auto_m 温和三态）
    score -f 文件.txt -v       附明细（AI 类概率 / 前三类 / 体裁判定）
    score -f 文件.txt -m lit   指定体裁单头判分
score -t "文本"            直接给文本（引号内用反斜杠 n 表示换行）
    use <模型>                 切换本次默认模型（如 use auto_p）
    -retry <模型>              用新模型重测**最近一次**（如 -retry auto_p）
    batch -i 输入.jsonl -o 输出.jsonl     批量判定

  ── 分析（不给概率，只给人话）──
    analysis -f 文件.txt       人话诊断：最像 AI / 最像人类的特征 + 改进建议
    analysis -f 文件.txt --seg 20 --top 3  长文分段分别诊断
    research -f 文件.txt -s    长文切片（按步长切块，逐块给体裁 + 人机判定）
    research -f 文件.txt -c    长文深度研究（滑窗 + 整合：主体裁/副体裁/概率）
                               加 -q 显示每行开头与每个窗口的首尾

  ── 本地微调 ──
    learn -f 我的诗.txt --genre lit --label human    标注并重训
    learn --list / --undo / --clear / --fit          查看 / 撤销 / 清空 / 重训
    reset                      清除本地微调，恢复内置模型

  ── 便捷模式（一句一句喂）──
    score -c                   进入；之后**直接粘贴文件路径 + 回车**即测
                               路径后可跟选项：--raw ｜ --no-seg ｜ -v ｜ -m lit
    score -e                   退出便捷模式

  ── 长文本模式（粘贴一大段）──
    score -l                   进入；粘贴文本，然后**另起一行输入 -run** 测量
                               -run 后可跟选项（如 -run --raw -m lit）
    -e                         取消（清空缓冲）
    score -e                   退出长文本模式

  ── 其他 ──
    info                       列出所有模型与当前设置
    help                       本帮助
    exit                       退出

  ── 模型与判定 ──
    auto_m（默认）温和：保守/激进一致才下结论，不一致报「不确定」
    auto_c 保守（宁可漏检也不误报）｜ auto_p 激进
    lit 诗 ｜ expo 说明文 ｜ narr 叙事 ｜ info 信息 ｜ classic 古文 ｜ applied 应用文 ｜ chat 对话
"""
