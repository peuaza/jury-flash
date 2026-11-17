# -*- coding: utf-8 -*-
"""score_v12.py — v1.2.0-rc 三态评分
  <=14 行：短诗头与 v23 一致才给 AI/人类，否则“不确定”
  >=15 行：v23 置信三态（p>=0.70 AI / p<0.30 人类 / 中间不确定）
"""
import argparse, io, json, os, sys, pickle
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import jury_det as JD
import cpu_proxy2 as CP

CFG=json.load(io.open(os.path.join(HERE,"jury_conf_v12.json"),encoding="utf-8"))
SH=pickle.load(io.open(os.path.join(HERE,CFG["short_head"]),"rb"))
_LIT=JD.Detector(path=os.path.join(HERE,"_det_lit_np.pkl"),model="__x__"); CP.load_lm()
_GEN=JD.Detector(path=os.path.join(HERE,CFG["general_model"]),model="__x__")

def short_p(lines):
    v=_LIT._raw_feats(lines)
    if v is None: return None
    x=np.array([v.get(k,np.nan) for k in SH["keys"]],float)
    if not np.isfinite(x).all(): return None
    z=(x-SH["mean"])/SH["scale"]; return float(1/(1+np.exp(-(SH["coef"]@z+SH["intercept"]))))

def score(lines):
    L=[x.strip() for x in lines if str(x).strip()]; n=len(L)
    if n<6: return {"ok":False,"error":"行数太少（<6 行）","n_lines":n}
    pv,info=_GEN._score_auto(L)
    if not info: return {"ok":False,"error":"特征抽取失败","n_lines":n}
    g=float(info.get("p_ai_total",0.0)); g_ai=bool(info.get("is_ai"))
    if n<=14:
        s=short_p(L)
        if s is None: return {"ok":False,"error":"短诗头特征抽取失败","n_lines":n}
        band="6-9" if n<=9 else "10-14"; thr=float(CFG["short_thresholds"][band])
        s_ai=bool(s>=thr)
        if s_ai and g_ai: verdict="AI"
        elif (not s_ai) and (not g_ai): verdict="人类"
        else: verdict="不确定"
        return {"ok":True,"model":"short-head+v23","n_lines":n,"band":band,
                "p_short":round(s,4),"thr_short":thr,"p_general":round(g,4),
                "verdict":verdict,"is_ai":verdict=="AI","uncertain":verdict=="不确定","genre":info.get("genre")}
    ai=float(CFG["general_bands"]["ai"]); hu=float(CFG["general_bands"]["human"])
    verdict="AI" if g>=ai else ("人类" if g<hu else "不确定")
    return {"ok":True,"model":"general-v23","n_lines":n,"p_general":round(g,4),
            "verdict":verdict,"is_ai":verdict=="AI","uncertain":verdict=="不确定","genre":info.get("genre")}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--json",action="store_true")
    a=ap.parse_args()
    txt=sys.stdin.read() if a.path=="-" else io.open(a.path,encoding="utf-8",errors="replace").read()
    out=score(txt.replace("\r\n","\n").split("\n"))
    if a.json: print(json.dumps(out,ensure_ascii=False,indent=2))
    elif not out.get("ok"): print("判分失败：%s"%out.get("error"))
    else: print("%s ｜ %d 行 ｜ %s ｜ p=%s"%(out["model"],out["n_lines"],out["verdict"],
          out.get("p_short",out.get("p_general"))))

if __name__=="__main__": main()
