# -*- coding: utf-8 -*-
"""_composition_feats.py — poem-level organization features (no doctor, no AI labels)."""
import io,os,sys,json,re,collections,math
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); PR=r"C:\Users\shini\poetry_axis\external_call\poem_reading"
import jieba, jieba.posseg as pseg
PUN="，。！？；：、…—～·「」『』（）()《》〈〉“”‘’\"'!?.,;:"
END=set("。！？；…—")
CONTRAST=set("但 却 而 可是 然而 只是 不过 只是 偏偏 忽然 突然 于是 所以 因为 如果 即使".split())
PRON=set("我 你 他 她 它 我们 你们 他们 她们 它们 自己 谁 什么".split())
DEM=set("这 那 此 其 该 些 这里 那里 这样 那样".split())
MODAL=set("会 能 要 想 应该 必须 可以 可能 也许 已经 曾经 将 正在 着 了 过".split())
NEG=set("不 没 无 未 别 莫 非".split())
def charbig(s):
    s=re.sub(r"\s+","",s); return set(s[i:i+2] for i in range(len(s)-1)) if len(s)>1 else set()
def jac(a,b):
    return len(a&b)/max(1,len(a|b))
def cos(a,b):
    if not a or not b: return 0.0
    return len(a&b)/math.sqrt(len(a)*len(b))
def feats(lines,title=""):
    L=[x.strip() for x in lines if x.strip()]
    n=len(L); chars=[len(x) for x in L]; words=[list(jieba.cut(x)) for x in L]; pos=[["x"]*len(ws) for ws in words]
    stanzas=[]; cur=0
    for x in lines:
        if x.strip(): cur+=1
        elif cur: stanzas.append(cur); cur=0
    if cur: stanzas.append(cur)
    allw=[w for ws in words for w in ws]; allpos=[p for ps in pos for p in ps]
    # punctuation
    punct=[sum(1 for ch in x if ch in PUN) for x in L]
    endp=[1 if x and x[-1] in END else 0 for x in L]
    commap=[x.count("，") for x in L]
    q=[1 if ("？" in x or "?" in x) else 0 for x in L]
    excl=[1 if ("！" in x or "!" in x) else 0 for x in L]
    # line starts/ends
    firstw=[ws[0] if ws else "" for ws in words]; lastw=[ws[-1] if ws else "" for ws in words]
    firstpos=[ps[0] if ps else "" for ps in pos]; lastpos=[ps[-1] if ps else "" for ps in pos]
    # lexical sets
    ws_sets=[set(w for w in ws if len(w)>1) for ws in words]
    cb=[charbig(x) for x in L]
    adj=[cos(cb[i],cb[i+1]) for i in range(n-1)]
    jadj=[jac(ws_sets[i],ws_sets[i+1]) for i in range(n-1)]
    # title
    tset=set(w for w in jieba.cut(title) if len(w)>1); body=set(allw)
    # repetition
    def rep_ratio(k):
        grams=[tuple(allw[i:i+k]) for i in range(max(0,len(allw)-k+1))]
        return 1-len(set(grams))/max(1,len(grams))
    f={
      "comp_n_lines":n,"comp_stanza_n":len(stanzas),"comp_stanza_mean":np.mean(stanzas) if stanzas else 0,"comp_stanza_sd":np.std(stanzas) if stanzas else 0,
      "comp_chars_mean":np.mean(chars) if chars else 0,"comp_chars_sd":np.std(chars) if chars else 0,"comp_chars_cv":np.std(chars)/max(1e-9,np.mean(chars)) if chars else 0,
      "comp_chars_min":min(chars) if chars else 0,"comp_chars_max":max(chars) if chars else 0,
      "comp_punct_mean":np.mean(punct) if punct else 0,"comp_punct_density":sum(punct)/max(1,sum(chars)),
      "comp_endstop":np.mean(endp) if endp else 0,"comp_enjamb":1-np.mean(endp) if endp else 0,
      "comp_comma_mean":np.mean(commap) if commap else 0,"comp_question":np.mean(q) if q else 0,"comp_exclaim":np.mean(excl) if excl else 0,
      "comp_first_diversity":len(set(firstw))/max(1,n),"comp_last_diversity":len(set(lastw))/max(1,n),
      "comp_first_pron":sum(1 for x in firstw if x in PRON)/max(1,n),"comp_first_conj":sum(1 for x in firstw if x in CONTRAST)/max(1,n),
      "comp_first_noun":sum(1 for x in firstpos if x.startswith("n"))/max(1,n),"comp_first_verb":sum(1 for x in firstpos if x.startswith("v"))/max(1,n),
      "comp_pron_density":sum(1 for w in allw if w in PRON)/max(1,len(allw)),"comp_dem_density":sum(1 for w in allw if w in DEM)/max(1,len(allw)),
      "comp_modal_density":sum(1 for w in allw if w in MODAL)/max(1,len(allw)),"comp_neg_density":sum(1 for w in allw if w in NEG)/max(1,len(allw)),
      "comp_contrast_density":sum(1 for w in allw if w in CONTRAST)/max(1,len(allw)),
      "comp_adj_cos_mean":np.mean(adj) if adj else 0,"comp_adj_cos_sd":np.std(adj) if adj else 0,"comp_adj_cos_min":min(adj) if adj else 0,"comp_adj_cos_max":max(adj) if adj else 0,
      "comp_adj_jac_mean":np.mean(jadj) if jadj else 0,"comp_adj_jac_sd":np.std(jadj) if jadj else 0,
      "comp_first_last_cos":cos(cb[0],cb[-1]) if n>1 else 0,"comp_half_cos":cos(set().union(*cb[:n//2]) if n//2 else set(),set().union(*cb[n//2:])),
      "comp_line_dispersion":np.mean([cos(cb[i],set().union(*cb)) for i in range(n)]) if n else 0,
      "comp_title_body_jac":jac(tset,body) if tset else 0,"comp_title_len":len(title),"comp_title_colon":1 if ("：" in title or ":" in title) else 0,
      "comp_lex_div":len(set(allw))/max(1,len(allw)),"comp_rep2":rep_ratio(2) if len(allw)>=2 else 0,"comp_rep3":rep_ratio(3) if len(allw)>=3 else 0,
      "comp_last_len":chars[-1] if chars else 0,"comp_last_q":q[-1] if q else 0,"comp_last_neg":sum(1 for w in words[-1] if w in NEG) if words else 0,
      "comp_last_novel":1-cos(cb[-1],set().union(*cb[:-1])) if n>1 else 0,
    }
    return f
def load_docs():
    out=[]
    for tag,base in (("拟目",os.path.join(PR,"_full_corpus")),("诗人",os.path.join(PR,"_poet_corpus"))):
        for fn in sorted(os.listdir(base)):
            if not fn.endswith(".txt"): continue
            L=[x.strip() for x in io.open(os.path.join(base,fn),encoding="utf-8",errors="replace") if x.strip()]
            if L: out.append((tag,fn[:-4],L))
    for fn,tag in (("_imitation3.jsonl","外部"),("_mimo_long_new.jsonl","MiMo")):
        for line in io.open(os.path.join(HERE,fn),encoding="utf-8"):
            d=json.loads(line); L=[str(x).strip() for x in d.get("lines",[]) if str(x).strip()]
            if L: out.append((tag,d.get("title") or d.get("theme") or "",L))
    return out
if __name__=="__main__":
    docs=load_docs(); keys=list(feats(["我 在 这里"],"x").keys())
    X=np.array([[feats(L,t)[k] for k in keys] for tag,t,L in docs],float)
    tags=np.array([tag for tag,_,_ in docs]); np.savez_compressed(os.path.join(HERE,"_composition_feats.npz"),X=X,tags=tags,keys=keys,names=np.array([t for _,t,_ in docs]))
    print("docs",len(docs),collections.Counter(tags),len(keys))
