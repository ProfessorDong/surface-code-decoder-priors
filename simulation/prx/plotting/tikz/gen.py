"""Turn captured matplotlib series into publication-quality pgfplots figures."""
import os
import tempfile
import json, math, re
import numpy as np
from collections import OrderedDict
from pathlib import Path

MAN = Path(os.environ.get("MANUSCRIPT_DIR", Path(__file__).resolve().parents[4] / "manuscript"))
DAT, FIG = MAN/"figdata", MAN/"figures"
cap = json.load(open(os.path.join(tempfile.gettempdir(), "tikz_captured.json")))

LS = {"-":"solid","--":"dashed",":":"dotted","-.":"dashdotted","none":"only marks","":"solid"}
MK = {"o":"*","s":"square*","^":"triangle*","v":"triangle*","D":"diamond*","":"none"}
LOC = {"upper left":"north west","upper right":"north east","lower left":"south west",
       "lower right":"south east","center left":"west","center right":"east","best":"north east",
       "lower center":"south","upper center":"north"}
COLORS = OrderedDict()
def cname(h):
    hx=(h or "").lstrip("#")
    if not re.fullmatch("[0-9a-fA-F]{6}",hx or ""):
        return {"white":"white","black":"black","gray":"black!45","grey":"black!45"}.get((h or "").lower(),"black")
    COLORS.setdefault(hx.upper(),"C%d"%len(COLORS)); return COLORS[hx.upper()]

def esc(s):
    s=(s or "").replace("\n"," ")
    s=re.sub(r"(?<!\\)%",r"\\%",s); s=re.sub(r"(?<!\\)&",r"\\&",s)
    return s
def tick_esc(s):
    s=(s or "").replace("\n","\\\\")
    s=re.sub(r"(?<!\\)%",r"\\%",s); return s
def ok(v): return v is not None and not (isinstance(v,float) and math.isnan(v))

def write_dat(name, cols, rows):
    with open(DAT/name,"w") as f:
        f.write(" ".join(cols)+"\n")
        for r in rows:
            if all(ok(v) for v in r): f.write(" ".join("%.10g"%v for v in r)+"\n")
    return name

def merge_scatter(series):
    out,grp=[],OrderedDict()
    for s in series:
        if s["kind"]=="line" and s.get("ls")=="none" and len(s.get("x",[]))<=2 and not s.get("label"):
            # the face colour is part of the key: open and filled markers must not merge
            grp.setdefault((s.get("color"),s.get("marker"),s.get("mfc","")),[]).extend(zip(s["x"],s["y"]))
        else: out.append(s)
    for (c,m,mfc),pts in grp.items():
        out.append({"kind":"line","x":[p[0] for p in pts],"y":[p[1] for p in pts],
                    "label":"","color":c,"marker":m,"mfc":mfc,"ls":"none"})
    return out

def axis_opts(r,w,h,single=False,mode=""):
    p=r["props"]; o=["width=%s"%w,"height=%s"%h]
    if p.get("xscale")=="log": o+=["xmode=log","log ticks with fixed point"]
    if p.get("yscale")=="log": o+=["ymode=log","log ticks with fixed point"]
    if p.get("title"): o.append("title={%s}"%esc(p["title"]))
    if p.get("xlabel"): o.append("xlabel={%s}"%esc(p["xlabel"]))
    if p.get("ylabel"): o.append("ylabel={%s}"%esc(p["ylabel"]))
    for lim,a,b in (("xlim","xmin","xmax"),("ylim","ymin","ymax")):
        v=p.get(lim)
        if isinstance(v,list) and len(v)==2 and all(ok(t) for t in v):
            o+=["%s=%.6g"%(a,v[0]),"%s=%.6g"%(b,v[1])]
    if p.get("xscale")=="log" and not p.get("xtick"):
        xs=sorted({v for ss in r["series"] for v in ss.get("x",[]) if ok(v)})
        span = (np.log10(max(xs)) - np.log10(min(xs))) if xs and min(xs) > 0 else 9
        if 0 < len(xs) <= 8 and span < 2.0:
            o.append("xtick={%s}"%",".join("%.6g"%v for v in xs))
    for t,l,style in (("xtick","xticklabels","x tick label style"),("ytick","yticklabels","y tick label style")):
        if p.get(t):
            o.append("%s={%s}"%(t,",".join("%.6g"%v for v in p[t] if ok(v))))
            if p.get(l):
                o.append("%s={%s}"%(l,",".join("{%s}"%tick_esc(x) for x in p[l])))
                if any("\n" in x for x in p[l]): o.append("%s={align=center}"%style)
    nlab=(sum(1 for ss in r["series"] if ss.get("label") and not ss["label"].startswith("_"))
          + sum(len(g) for g in r.get("legend_handles", [])))
    if mode == "shared":
        o.append("legend to name=leg%s" % re.sub(r"[^A-Za-z]", "", r.get("figure","")))
        o.append("legend columns=%d" % min(max(nlab, 1), 4))
        o.append("legend style={draw=black!30,font=%s,inner sep=1.5pt,column sep=4pt,legend cell align=left}"
                 % LEGEND_FONT.get(r.get("figure"), "\\tiny"))
    elif mode == "below" and r.get("figure") in BELOW_LEFT:
        # a legend wider than its panel: anchor it at the panel's left edge so it stays inside the figure
        o.append("legend style={at={(-0.18,-0.32)},anchor=north west,draw=black!30,"
                 "font=%s,inner sep=1.2pt,row sep=-1.5pt,legend cell align=left}"
                 % LEGEND_FONT.get(r.get("figure"), "\\tiny"))
        o.append("legend columns=2")
    elif mode == "below":
        o.append("legend style={at={(0.5,-0.32)},anchor=north,draw=black!30,"
                 "font=%s,inner sep=1.2pt,row sep=-1.5pt,legend cell align=left}"
                 % LEGEND_FONT.get(r.get("figure"), "\\tiny"))
        o.append("legend columns=2")
    elif nlab > 5 and single and r.get("figure") not in INSIDE_LEGEND:
        o.append("legend style={at={(0.5,-0.30)},anchor=north,draw=black!30,"
                 "font=%s,inner sep=1.2pt,row sep=-1.5pt,legend cell align=left}"
                 % LEGEND_FONT.get(r.get("figure"), "\\tiny"))
        o.append("legend columns=1")
    else:
        pos = LOC.get(p.get("legend_loc","best"),"north east")
        if pos == "south":
            o.append("legend style={at={(0.5,0.03)},anchor=south}")
        elif pos == "north":
            o.append("legend style={at={(0.5,0.97)},anchor=north}")
        else:
            o.append("legend pos=%s"%pos)
        if int(p.get("legend_ncol",1) or 1)>1:
            o.append("legend columns=%d"%int(p["legend_ncol"]))
        if r.get("figure") in LEGEND_FONT:
            o.append("legend style={font=%s}" % LEGEND_FONT[r["figure"]])
        if r.get("figure") in INSIDE_LEGEND:
            o.append("legend style={font=%s,draw=black!40,fill=white,fill opacity=%.2f,draw opacity=1,"
                     "text opacity=1,inner sep=1.5pt,column sep=3pt,row sep=-1pt}"
                     % (INSIDE_LEGEND[r["figure"]], LEGEND_OPACITY.get(r["figure"], 0.9)))
    return o

def xrange(r):
    lim=r["props"].get("xlim")
    if isinstance(lim,list) and len(lim)==2: return lim[0],lim[1]
    xs=[v for s in r["series"] for v in s.get("x",[]) if ok(v)]
    return (min(xs),max(xs)) if xs else (0,1)

PLAN = {}
# Figures whose legends are drawn inside each panel, where the plotting code puts
# them, in a box and at a slightly larger size than the default \tiny.
INSIDE_LEGEND = {"fig1_prior_geometry": "\\scriptsize", "fig2_exact_theory": "\\scriptsize",
                 "fig5_prior_vs_decoder": "\\scriptsize"}
# fill opacity of inside legend boxes (default 0.9); lower is more translucent
LEGEND_OPACITY = {"fig5_prior_vs_decoder": 0.6}
# Figures whose legends (shared or inside) are set one size up from \\tiny.
BELOW_LEFT = {"fig6_lambda"}
LEGEND_FONT = {"fig3_heterogeneity": "\\scriptsize", "fig4_willow_frontier": "\\scriptsize", "fig6_lambda": "\\scriptsize"}
def labels_of(r):
    labs = [ss.get("label","") for ss in r["series"] if ss.get("label")]
    labs += [h.get("label","") for g in r.get("legend_handles", []) for h in g]
    return frozenset(l for l in labs if l and not l.startswith("_"))

def plan_legends(fig, axes):
    sets = [labels_of(r) for r in axes]
    nonempty = [i for i, x in enumerate(sets) if x]
    if fig in INSIDE_LEGEND:
        for i in nonempty:
            PLAN[(fig, i)] = "inside"
        return
    if len(axes) > 1 and len(nonempty) >= 2:
        big = max(nonempty, key=lambda i: len(sets[i]))
        group = [i for i in nonempty if sets[i] <= sets[big]]
        if len(group) >= 2:
            # panels repeating one legend share a single copy below the figure;
            # a panel with a different legend keeps its own, inside the axes
            for i in nonempty:
                PLAN[(fig, i)] = ("shared" if i == big else "suppress") if i in group else "inside"
            return
    for i in nonempty:
        PLAN[(fig, i)] = "below" if (len(axes) > 1 and len(sets[i]) > 4) else "inside"

SEEN_LEGENDS = {}
def legend_key(r):
    labs = [ss.get("label","") for ss in r["series"] if ss.get("label")]
    labs += [h.get("label","") for g in r.get("legend_handles", []) for h in g]
    return (r.get("figure"), tuple(sorted(set(l for l in labs if l and not l.startswith("_")))))

def emit(r,tag,single=False):
    out=[]; series=merge_scatter(r["series"]); x0,x1=xrange(r); k=0
    suppress = PLAN.get((r.get("figure"), r.get("_panel")), "") == "suppress"
    has_handles = bool(r.get("legend_handles")) or suppress
    for s in series:
        k+=1; c=cname(s.get("color")); lab=s.get("label",""); kind=s["kind"]
        leg="\n\\addlegendentry{%s}"%esc(lab) if (lab and not has_handles) else ""
        if kind=="line":
            if not s.get("x"): continue
            f=write_dat("%s_s%d.dat"%(tag,k),["x","y"],list(zip(s["x"],s["y"])))
            openm = s.get("mfc","") in ("none","None")
            if s.get("ls")=="none":
                sty=[c,"only marks","mark=%s"%MK.get(s.get("marker",""),"*"),"mark size=1.3pt"]
                if openm: sty.append("mark options={fill=white}")
            else:
                sty=[c,LS.get(s.get("ls","-"),"solid"),"mark=%s"%MK.get(s.get("marker",""),"none"),
                     "mark size=1.4pt","line width=0.7pt","mark options={solid}"]
            if not leg: sty.append("forget plot")
            out.append("\\addplot[%s] table[x=x,y=y] {figdata/%s};%s"%(",".join(sty),f,leg))
        elif kind=="errorbar":
            if not s.get("x"): continue
            es=s.get("yerr") or [0.0]*len(s["x"])
            asym = len(es) == 2 * len(s["x"])        # matplotlib [[minus...], [plus...]] flattened
            if asym:
                f=write_dat("%s_s%d.dat"%(tag,k),["x","y","em","ep"],
                            list(zip(s["x"],s["y"],es[:len(s["x"])],es[len(s["x"]):])))
                tbl="table[x=x,y=y,y error plus=ep,y error minus=em]"
            else:
                f=write_dat("%s_s%d.dat"%(tag,k),["x","y","e"],list(zip(s["x"],s["y"],es)))
                tbl="table[x=x,y=y,y error=e]"
            none_ls = s.get("ls") in ("none","None","")
            sty=[c,"only marks" if none_ls else LS.get(s.get("ls","-"),"solid"),"mark=%s"%MK.get(s.get("marker","o"),"*"),
                 "mark size=1.3pt","line width=0.7pt","mark options={solid}","error bars/.cd","y dir=both","y explicit",
                 "error bar style={line width=0.4pt}"]
            if not leg: sty.insert(0, "forget plot")
            out.append("\\addplot[%s] %s {figdata/%s};%s"%(",".join(sty),tbl,f,leg))
        elif kind=="band":
            if not s.get("x"): continue
            f=write_dat("%s_s%d.dat"%(tag,k),["x","lo","hi"],list(zip(s["x"],s["y1"],s["y2"])))
            nm="%sB%d"%(re.sub(r"[^A-Za-z]","",tag),k)
            out+=["\\addplot[name path=%sL,draw=none,forget plot] table[x=x,y=lo] {figdata/%s};"%(nm,f),
                  "\\addplot[name path=%sH,draw=none,forget plot] table[x=x,y=hi] {figdata/%s};"%(nm,f),
                  "\\addplot[%s,fill opacity=%.2f,forget plot] fill between[of=%sL and %sH];"
                  %(c,min(s.get("alpha") or 0.25,0.30),nm,nm)]
        elif kind=="bar":
            if not s.get("x"): continue
            ye = s.get("yerr") or []
            if len(ye) == 2 * len(s["x"]):          # matplotlib [[minus...], [plus...]] flattened
                m_, p_ = ye[:len(s["x"])], ye[len(s["x"]):]
                f=write_dat("%s_s%d.dat"%(tag,k),["x","y","em","ep"],list(zip(s["x"],s["h"],m_,p_)))
                ebar = (",error bars/.cd,y dir=both,y explicit,error bar style={line width=0.5pt,black!75},"
                        "error mark options={rotate=90,mark size=1.2pt,line width=0.5pt,black!75}")
                tbl = "table[x=x,y=y,y error plus=ep,y error minus=em]"
            else:
                f=write_dat("%s_s%d.dat"%(tag,k),["x","y"],list(zip(s["x"],s["h"])))
                ebar, tbl = "", "table[x=x,y=y]"
            edge = cname(s.get("edgecolor")) if s.get("edgecolor") else "black!55"
            if c == "white":
                fill = "fill=white,draw=%s" % edge
                if s.get("hatch"):
                    fill += ",postaction={pattern=north east lines,pattern color=%s}" % edge
            else:
                fill = "fill=%s,draw=%s" % (c, edge)
            out.append("\\addplot[ybar,bar width=6pt,line width=0.4pt,%s,forget plot%s] "
                       "%s {figdata/%s};"%(fill,ebar,tbl,f))
        elif kind=="hline":
            labelled = bool(leg) and not lab.startswith("_")
            col = "black!70" if labelled else "black!45"
            out.append("\\addplot[%s,%s,line width=%s%s] coordinates {(%.10g,%.10g) (%.10g,%.10g)};%s"
                       %(col,LS.get(s.get("ls","--"),"dashed"),"0.6pt" if labelled else "0.5pt","" if labelled else ",forget plot",
                         x0,s.get("y",0),x1,s.get("y",0),leg if labelled else ""))
    for group in ([] if suppress else r.get("legend_handles", [])):
        for h in group:
            if not h.get("label") or h["label"].startswith("_"): continue
            if h.get("kind") == "patch":
                fc, ec = cname(h.get("facecolor")), cname(h.get("edgecolor"))
                img = "area legend,fill=%s,draw=%s" % (fc, ec)
                if h.get("hatch"):
                    img += ",postaction={pattern=north east lines,pattern color=%s}" % ec
            else:
                c = cname(h.get("color")); m = MK.get(h.get("marker",""), "none")
                ls = h.get("ls","-")
                img = "%s,%s,mark=%s,mark size=1.4pt" % (c, "only marks" if ls in ("None","none","") else LS.get(ls, "solid"), m)
                if h.get("mfc") == "none": img += ",mark options={fill=white}"
            out.append("\\addlegendimage{%s}\n\\addlegendentry{%s}" % (img, esc(h["label"])))
    for t in ([] if single else r.get("texts", [])):
        if t.get("axes") and re.fullmatch(r"\([a-z]\)", t["text"].strip()):
            out.append("\\node[anchor=south west,font=\\small\\bfseries] at (rel axis cs:0,1.06) {%s};"
                       % (t["text"],))
    return out

byfig=OrderedDict()
for r in cap: byfig.setdefault(r.get("figure","fig"),[]).append(r)
GEOM={1:("0.68\\columnwidth","4.0cm",1), 2:("0.37\\textwidth","4.2cm",2),
      3:("0.24\\textwidth","3.8cm",3),   4:("0.37\\textwidth","4.0cm",2)}
for fig,axes in byfig.items():
    w,h,ncol=GEOM.get(len(axes),("0.40\\textwidth","4.0cm",2))
    nrow=max(1,(len(axes)+ncol-1)//ncol)
    vsep = "2.1cm" if (nrow > 1 and any(r["props"].get("title") for r in axes)) else "1.45cm"
    P=["\\begin{tikzpicture}",
       "\\begin{groupplot}[group style={group size=%d by %d, horizontal sep=1.35cm, vertical sep=%s},prxaxis]"%(ncol,nrow,vsep)]
    plan_legends(fig, axes)
    for i,r in enumerate(axes):
        r["_panel"] = i
        mode = PLAN.get((fig, i), "")
        P.append("\\nextgroupplot[%s]"%",".join(axis_opts(r,w,h,single=(len(axes)==1),mode=mode)))
        P+=emit(r,"%s_p%d"%(fig,i),single=(len(axes)==1))
    P.append("\\end{groupplot}")
    if any(PLAN.get((fig, i)) == "shared" for i in range(len(axes))):
        P.append("\\node[anchor=north] at ([yshift=-2pt]current bounding box.south) "
                 "{\\pgfplotslegendfromname{leg%s}};" % re.sub(r"[^A-Za-z]", "", fig))
    P.append("\\end{tikzpicture}")
    (FIG/("%s.tex"%fig)).write_text("\n".join(P)+"\n")
    print("figures/%s.tex  %d panel(s)"%(fig,len(axes)))

sty=["% generated by tikz/gen.py -- shared pgfplots style",
     "\\usepgfplotslibrary{groupplots,fillbetween}","\\usetikzlibrary{patterns}","\\pgfplotsset{compat=1.18}"]
for h_,n in COLORS.items(): sty.append("\\definecolor{%s}{HTML}{%s}"%(n,h_))
sty.append(r"""\pgfplotsset{prxaxis/.style={
  scale only axis, clip mode=individual, axis lines=left, axis line style={black!70, line width=0.4pt},
  tick align=outside, tick style={black!70, line width=0.4pt},
  label style={font=\footnotesize}, tick label style={font=\scriptsize},
  legend style={font=\tiny, draw=black!30, fill=white, fill opacity=0.88,
                text opacity=1, inner sep=1.2pt, row sep=-1.5pt, legend cell align=left},
  grid=major, grid style={black!10, line width=0.3pt},
  every axis plot/.append style={line join=round, line cap=round},
}}""")
(MAN/"prxfig.tex").write_text("\n".join(sty)+"\n")
print("prxfig.tex: %d colors"%len(COLORS))
