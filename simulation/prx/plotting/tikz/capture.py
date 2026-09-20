"""Run the existing matplotlib figure code with Axes instrumented, and dump
every plotted series to JSON so it can be re-rendered in pgfplots."""
import os
from pathlib import Path
import tempfile
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

REC = {}          # id(ax) -> dict
ORDER = []        # axes in creation order

def _ax(ax):
    k = id(ax)
    if k not in REC:
        REC[k] = {"series": [], "props": {}}
        ORDER.append(k)
    return REC[k]

def _arr(x):
    a = np.asarray(x, dtype=object)
    try: return [None if v is None else float(v) for v in np.ravel(a)]
    except Exception: return [str(v) for v in np.ravel(a)]

_DEPTH = [0]   # >0 while inside a wrapped call: nested calls (bar -> errorbar) are internal

def wrap(name, kind, argmap):
    orig = getattr(Axes, name)
    def f(self, *a, **kw):
        if _DEPTH[0] == 0:
            try: _ax(self)["series"].append(dict(kind=kind, **argmap(a, kw)))
            except Exception as e: _ax(self)["series"].append(dict(kind=kind, error=str(e)))
        _DEPTH[0] += 1
        try:
            return orig(self, *a, **kw)
        finally:
            _DEPTH[0] -= 1
    setattr(Axes, name, f)

wrap("plot","line", lambda a,kw: dict(
    x=_arr(a[0]) if len(a)>0 else [], y=_arr(a[1]) if len(a)>1 else [],
    label=str(kw.get("label","")), color=str(kw.get("color","")),
    ls=str(kw.get("linestyle",kw.get("ls","-"))), marker=str(kw.get("marker","")),
    mfc=str(kw.get("markerfacecolor",kw.get("mfc",""))),
    lw=kw.get("linewidth",kw.get("lw",None))))
wrap("errorbar","errorbar", lambda a,kw: dict(
    x=_arr(a[0]) if len(a)>0 else [], y=_arr(a[1]) if len(a)>1 else [],
    yerr=_arr(kw["yerr"]) if kw.get("yerr") is not None else [],
    label=str(kw.get("label","")), color=str(kw.get("color","")),
    marker=str(kw.get("marker","")), ls=str(kw.get("linestyle",kw.get("ls","-")))))
wrap("fill_between","band", lambda a,kw: dict(
    x=_arr(a[0]) if len(a)>0 else [], y1=_arr(a[1]) if len(a)>1 else [],
    y2=_arr(a[2]) if len(a)>2 else [], color=str(kw.get("color","")),
    alpha=kw.get("alpha",None), label=str(kw.get("label",""))))
wrap("bar","bar", lambda a,kw: dict(
    x=_arr(a[0]) if len(a)>0 else [], h=_arr(a[1]) if len(a)>1 else [],
    label=str(kw.get("label","")), color=str(kw.get("color","")),
    edgecolor=str(kw.get("edgecolor","")), hatch=str(kw.get("hatch","") or ""),
    yerr=_arr(kw["yerr"]) if kw.get("yerr") is not None else []))
wrap("scatter","scatter", lambda a,kw: dict(
    x=_arr(a[0]) if len(a)>0 else [], y=_arr(a[1]) if len(a)>1 else [],
    label=str(kw.get("label","")), color=str(kw.get("color",kw.get("c","")))))
wrap("axhline","hline", lambda a,kw: dict(y=float(a[0]) if a else float(kw.get("y",0)),
    ls=str(kw.get("linestyle",kw.get("ls","-"))), color=str(kw.get("color","")),
    label=str(kw.get("label",""))))
wrap("axvline","vline", lambda a,kw: dict(x=float(a[0]) if a else float(kw.get("x",0)),
    ls=str(kw.get("linestyle",kw.get("ls","-"))), color=str(kw.get("color","")),
    label=str(kw.get("label",""))))

for prop, key in [("set_xlabel","xlabel"),("set_ylabel","ylabel"),("set_title","title"),
                  ("set_xscale","xscale"),("set_yscale","yscale")]:
    def mk(prop=prop,key=key):
        orig=getattr(Axes,prop)
        def f(self,*a,**kw):
            _ax(self)["props"][key]=str(a[0]) if a else ""
            return orig(self,*a,**kw)
        return f
    setattr(Axes,prop,mk())
# tick positions and labels, and legend placement
for prop,key in [("set_xticks","xtick"),("set_yticks","ytick")]:
    def mk3(prop=prop,key=key):
        orig=getattr(Axes,prop)
        def f(self,*a,**kw):
            try: _ax(self)["props"][key]=_arr(a[0]) if a else []
            except Exception: pass
            if kw.get("labels") is not None:
                _ax(self)["props"][key+"labels"]=[str(v) for v in kw["labels"]]
            return orig(self,*a,**kw)
        return f
    setattr(Axes,prop,mk3())
for prop,key in [("set_xticklabels","xticklabels"),("set_yticklabels","yticklabels")]:
    def mk4(prop=prop,key=key):
        orig=getattr(Axes,prop)
        def f(self,*a,**kw):
            try: _ax(self)["props"][key]=[str(getattr(v,"get_text",lambda:v)()) for v in (a[0] if a else [])]
            except Exception: pass
            return orig(self,*a,**kw)
        return f
    setattr(Axes,prop,mk4())
_origleg = Axes.legend
def _leg(self,*a,**kw):
    _ax(self)["props"]["legend_loc"]=str(kw.get("loc","best"))
    _ax(self)["props"]["legend_ncol"]=int(kw.get("ncol",1) or 1)
    return _origleg(self,*a,**kw)
Axes.legend=_leg


# ---- titles, panel letters, and legends built from proxy handles
_origtitle = Axes.set_title
def _title(self, *a, **kw):
    _ax(self)["props"]["title"] = str(a[0]) if a else ""
    return _origtitle(self, *a, **kw)
Axes.set_title = _title

_origtext = Axes.text
def _text(self, x, y, t, *a, **kw):
    try:
        tr = kw.get("transform", None)
        _ax(self).setdefault("texts", []).append(dict(x=float(x), y=float(y), text=str(t),
            axes=(tr is self.transAxes), ha=str(kw.get("ha","left")), va=str(kw.get("va","baseline")),
            bold=str(kw.get("fontweight","")) == "bold"))
    except Exception:
        pass
    return _origtext(self, x, y, t, *a, **kw)
Axes.text = _text

def _handle(h):
    from matplotlib.colors import to_hex
    d = {"label": str(h.get_label())}
    try:
        d["kind"] = "patch" if h.__class__.__name__ in ("Patch", "Rectangle") else "line"
        if d["kind"] == "patch":
            d["facecolor"] = to_hex(h.get_facecolor(), keep_alpha=False)
            d["edgecolor"] = to_hex(h.get_edgecolor(), keep_alpha=False)
            d["hatch"] = h.get_hatch() or ""
        else:
            d["color"] = to_hex(h.get_color(), keep_alpha=False)
            d["marker"] = str(h.get_marker())
            d["ls"] = str(h.get_linestyle())
            mfc = h.get_markerfacecolor()
            d["mfc"] = "none" if (isinstance(mfc, str) and mfc == "none") else to_hex(mfc, keep_alpha=False)
    except Exception as e:
        d["error"] = str(e)
    return d

_origleg2 = Axes.legend
def _leg2(self, *a, **kw):
    rec = _ax(self)
    if kw.get("handles") is not None:
        rec.setdefault("legend_handles", []).append([_handle(h) for h in kw["handles"]])
    rec["props"]["legend_loc"] = str(kw.get("loc", "best"))
    rec["props"]["legend_ncol"] = int(kw.get("ncol", 1) or 1)
    return _origleg2(self, *a, **kw)
Axes.legend = _leg2

for prop,key in [("set_xlim","xlim"),("set_ylim","ylim")]:
    def mk2(prop=prop,key=key):
        orig=getattr(Axes,prop)
        def f(self,*a,**kw):
            try: _ax(self)["props"][key]=_arr(a[0]) if len(a)==1 else [float(a[0]),float(a[1])]
            except Exception: pass
            return orig(self,*a,**kw)
        return f
    setattr(Axes,prop,mk2())

_origsave = plt.Figure.savefig
def savefig(self, path, *a, **kw):
    name = str(path).split("/")[-1].replace(".pdf","")
    for k in list(ORDER):
        if k in REC and "figure" not in REC[k]:
            REC[k]["figure"] = name
    return _origsave(self, path, *a, **kw)
plt.Figure.savefig = savefig

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # repository root
_ARGS = sys.argv[1:]
sys.argv = ["figures"]
from simulation.prx.plotting.figures import main
main(_ARGS)

out = []
for i,k in enumerate(ORDER):
    r = REC[k]; r["axis_index"] = i
    out.append(r)
json.dump(out, open(os.path.join(tempfile.gettempdir(), "tikz_captured.json"), "w"), indent=1)
print("captured %d axes across %d figures" % (len(out), len({r.get('figure') for r in out})))
