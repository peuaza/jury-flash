# -*- coding: utf-8 -*-
"""score_v11rc.py — v1.1.0-rc 评分入口（分长度档阈值）

用法:
  python score_v11rc.py poem.txt
  python score_v11rc.py poem.txt --json
  python score_v11rc.py -            # stdin

说明:
  - 加载 `_det_multi14_v22d.pkl`（定向仿作修复候选）；
  - 判定 = p_ai_total >= threshold[行数档]，阈值来自 jury_conf_v11rc.json（FP5%）；
  - **不修改默认模型**；正式版仍用 _det_multi14.pkl。
"""
import argparse, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jury_det as JD  # noqa: E402

CFG = json.load(io.open(os.path.join(HERE, "jury_conf_v11rc.json"), encoding="utf-8"))
MODEL = os.path.join(HERE, CFG["model"])


def band(n):
    return "6-7" if n <= 7 else "8-9" if n <= 9 else "10-14" if n <= 14 else "15+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="文本文件，或 - 表示 stdin")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    txt = sys.stdin.read() if a.path == "-" else io.open(a.path, encoding="utf-8", errors="replace").read()
    lines = [x.strip() for x in txt.replace("\r\n", "\n").split("\n") if x.strip()]
    n = len(lines)
    if n < 6:
        out = {"ok": False, "error": "行数太少（<6 行）", "n_lines": n}
    else:
        D = JD.Detector(path=MODEL, model="__x__")
        pv, info = D._score_auto(lines)
        if not info:
            out = {"ok": False, "error": "特征抽取失败", "n_lines": n}
        else:
            p = float(info.get("p_ai_total", 0.0))
            thr = float(CFG["thresholds"][band(n)])
            out = {"ok": True, "n_lines": n, "band": band(n),
                   "p_ai_total": round(p, 4), "threshold": thr,
                   "is_ai": bool(p >= thr - 1e-9),
                   "genre": info.get("genre"), "version": CFG["version"]}
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        if not out.get("ok"):
            print("判分失败：%s（%d 行）" % (out.get("error"), out.get("n_lines", 0)))
        else:
            print("%s ｜ %d 行（%s 档）｜ p_ai_total=%.4f ≥ %.3f ⇒ **%s** ｜ 体裁 %s"
                  % (out["version"], out["n_lines"], out["band"], out["p_ai_total"],
                     out["threshold"], "AI" if out["is_ai"] else "人类", out["genre"]))


if __name__ == "__main__":
    main()
