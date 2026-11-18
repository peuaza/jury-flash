# -*- coding: utf-8 -*-
"""score_v14.py — v1.4 candidate: 206 poem base + 46-dim author head."""
import io,os,sys,json,argparse
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE)); sys.path.insert(0,HERE)
import jury_det as JD
from composition_feats import feats as cfeats
BASE=os.path.join(HERE,"_poemguard_v4_206.pkl"); HEAD=os.path.join(HERE,"_composition_head_v4.npz")
W=np.load(HEAD,allow_pickle=True); B=__import__("pickle").load(io.open(BASE,"rb"))
D=JD.Detector(path=os.path.join(os.path.dirname(HERE),"_det_multi14.pkl"),model="__x__"); D.use_tok=True; D.use_cpu_surp=True
def score(lines,title=""):
    v=D._raw_feats([x.strip() for x in lines if x.strip()])
    if v is None: return {"error":"too short"}
    x160=np.array([v[k] for k in B["keys"][:160]],float)
    c=cfeats(lines,title); x46=np.array([c[k] for k in W["keys"]],float)
    x=np.r_[x160,x46]; z=(x-B["mean"])/B["scale"]; p=float(1/(1+np.exp(-(z@B["coef"]+B["intercept"]))))
    ch=(x46-W["mean"])/W["scale"]; cp=float(1/(1+np.exp(-(ch@W["coef"]+W["intercept"]))))
    if p < 0.298: pred="human"
    elif cp >= 0.7: pred="human"
    else: pred="ai"
    return {"pred":pred,"p_base":round(p,4),"p_author":round(cp,4),"n_lines":len(lines)}
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--file"); ap.add_argument("--title",default=""); a=ap.parse_args()
    L=[x.strip() for x in io.open(a.file,encoding="utf-8",errors="replace") if x.strip()]
    print(json.dumps(score(L,a.title),ensure_ascii=False))
