# -*- coding: utf-8 -*-
"""score_v18.py — v1.8: short 206 model for <=14; v1.7 base+head for >=15."""
import io,os,sys,json,argparse,pickle
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import jury_det as JD
from composition_feats import feats
D=JD.Detector(path=os.path.join(ROOT,"_det_multi14.pkl"),model="__x__"); D.use_tok=True; D.use_cpu_surp=True
B=pickle.load(io.open(os.path.join(HERE,"_poemguard_v7_206.pkl"),"rb")); H=np.load(os.path.join(HERE,"_composition_head_v7.npz"),allow_pickle=True); S=np.load(os.path.join(HERE,"_short_v20.npz"),allow_pickle=True)
def comp(lines,title): return feats(lines,title)
def score(lines,title=""):
 L=[x.strip() for x in lines if str(x).strip()]; n=len(L)
 if n<6: return {"error":"too short","n":n}
 v=D._raw_feats(L); c=comp(L,title)
 if n<=14:
  x=np.r_[[v[k] for k in S["keys"][:160]],[c[k] for k in S["keys"][160:]]]; z=(x-S["mean"])/S["scale"]; p=float(1/(1+np.exp(-(z@S["coef"]+S["intercept"])))); return {"pred":"AI" if p>=float(S["thr"]) else "human","p_short":round(p,4),"n":n}
 x=np.r_[[v[k] for k in B["keys"][:160]],[c[k] for k in B["keys"][160:]]]; z=(x-B["mean"])/B["scale"]; pb=float(1/(1+np.exp(-(z@B["coef"]+B["intercept"])))); ch=(x[160:]-H["mean"])/H["scale"]; ph=float(1/(1+np.exp(-(ch@H["coef"]+H["intercept"])))); return {"pred":"AI" if (pb>=0.292687 and ph<0.8) else "human","p_base":round(pb,4),"p_author":round(ph,4),"n":n}
if __name__=="__main__":
 ap=argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--title",default=""); a=ap.parse_args(); L=io.open(a.path,encoding="utf-8",errors="replace").read().splitlines(); print(json.dumps(score(L,a.title),ensure_ascii=False))
