# -*- coding: utf-8 -*-
"""§9.205 jury 配置：默认模型（持久化）
   位置：EXE 所在目录 / 当前目录 / ~/.jury/  （取第一个可写的）
"""
import io, json, os, sys

NAME = "jury_default.json"


def _dirs():
    ds = []
    if getattr(sys, "frozen", False):
        ds.append(os.path.dirname(os.path.abspath(sys.executable)))
    ds.append(os.path.dirname(os.path.abspath(__file__)))
    ds.append(os.getcwd())
    ds.append(os.path.join(os.path.expanduser("~"), ".jury"))
    return ds


def _cfg_path(write=False):
    if write:
        for d in _dirs():
            try:
                os.makedirs(d, exist_ok=True)
                t = os.path.join(d, "._wt")
                io.open(t, "w").close(); os.remove(t)
                return os.path.join(d, NAME)
            except Exception:
                continue
        return None
    for d in _dirs():
        p = os.path.join(d, NAME)
        if os.path.exists(p): return p
    return None


def get_default():
    p = _cfg_path(False)
    if not p: return None
    try:
        d = json.load(io.open(p, encoding="utf-8"))
        return d.get("model")
    except Exception:
        return None


def set_default(model):
    p = _cfg_path(True)
    if not p: return False
    try:
        io.open(p, "w", encoding="utf-8").write(json.dumps({"model": model}, ensure_ascii=False))
        return True
    except Exception:
        return False


def clear_default():
    n = 0
    for d in _dirs():
        p = os.path.join(d, NAME)
        if os.path.exists(p):
            try: os.remove(p); n += 1
            except Exception: pass
    return n
