# -*- coding: utf-8 -*-
"""Cross-identity reliability analysis for FER-Reliability-Benchmark.

Validated definitions (fixed):
  A: A_angular_distance_deg >= 13.43702602 for 3 consecutive viewpoints.
  C: C_margin <= 0.00237080 for 3 consecutive viewpoints.
  B: B_predicted_folder != true folder for 3 consecutive viewpoints.

Traversal is side-specific from frontal viewpoint 107:
  left  = 107 -> 106 -> ... -> 0
  right = 107 -> 108 -> ... -> 214

Lead is measured as distance-from-frontal(B) - distance-from-frontal(A).
Positive lead means A precedes B. No threshold is estimated here.
"""
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(r"D:\1405\FER-Reliability-Benchmark")
INPUT_FILE = PROJECT_ROOT / "analysis" / "4_analyze_embeddings_trajectory" / "per_view_metrics_multimetric.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "12_analyze_cross_identity"

A_THRESHOLD = 13.43702602
C_THRESHOLD = 0.00237080
SUSTAINED_VIEWPOINTS = 3
FRONTAL_VIEWPOINT = 107
MIN_VIEWPOINT = 0
MAX_VIEWPOINT = 214
EXPECTED_VIEWPOINTS = 215
HORIZONS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
MIN_PRIMARY_SEQUENCES = 5
BOOTSTRAP_REPS = 5000
PERMUTATION_REPS = 5000
SEED = 20260810

COL_EXPR = "expression"
COL_FOLDER = "folder"
COL_VIEW = "viewpoint"
COL_A = "A_angular_distance_deg"
COL_C = "C_margin"
COL_B = "B_predicted_folder"


def banner(s):
    print("\n" + "#" * 70)
    print(s)
    print("#" * 70)


def txt(x):
    return "" if pd.isna(x) else str(x).strip()


def num(s):
    return pd.to_numeric(s, errors="coerce")


def finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def json_safe(obj):
    """Recursively convert pandas/NumPy objects and non-finite floats to JSON-safe values."""
    if obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value=float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str,int,bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple,set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return json_safe(obj.tolist())
    return json_safe(str(obj))


def wilson(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0.0, c-h), min(1.0, c+h)


def bootstrap_mean(x, reps=5000, seed=SEED):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for i in range(reps):
        out[i] = np.mean(rng.choice(x, len(x), replace=True))
    return float(np.mean(x)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_perm(a, b, reps=5000, seed=SEED):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if len(d) == 0:
        return np.nan, np.nan
    obs = float(np.mean(d))
    rng = np.random.default_rng(seed)
    null = np.empty(reps)
    for i in range(reps):
        null[i] = np.mean(d * rng.choice([-1.0, 1.0], len(d)))
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (reps + 1)
    return obs, float(p)


def range_perm(groups, reps=5000, seed=SEED):
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups if len(g)]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    obs_rates = [np.mean(g) for g in groups]
    obs = float(max(obs_rates) - min(obs_rates))
    pooled = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    rng = np.random.default_rng(seed)
    null = np.empty(reps)
    for r in range(reps):
        sh = rng.permutation(pooled)
        rates, start = [], 0
        for n in sizes:
            rates.append(np.mean(sh[start:start+n])); start += n
        null[r] = max(rates) - min(rates)
    p = (np.sum(null >= obs) + 1) / (reps + 1)
    return obs, float(np.mean(null)), float(p)


def first_sustained(condition, n=3):
    condition = np.asarray(condition, dtype=bool)
    run = 0
    for i, v in enumerate(condition):
        run = run + 1 if v else 0
        if run >= n:
            return i - n + 1
    return None


def sustained_ge(values, threshold):
    v = np.asarray(values, dtype=float)
    return first_sustained(np.isfinite(v) & (v >= threshold), SUSTAINED_VIEWPOINTS)


def sustained_le(values, threshold):
    v = np.asarray(values, dtype=float)
    return first_sustained(np.isfinite(v) & (v <= threshold), SUSTAINED_VIEWPOINTS)


def sustained_B(pred, true_folder):
    pred = np.asarray([txt(x) for x in pred], dtype=object)
    mismatch = np.array([(x != txt(true_folder)) for x in pred], dtype=bool)
    return first_sustained(mismatch, SUSTAINED_VIEWPOINTS)


def parse_identity(folder):
    f = txt(folder)
    lo = f.lower()
    if lo.startswith("female"):
        return "Female", "Female_Base", "Female"
    if lo.startswith("male"):
        m = re.search(r"ZZ[_ ](.+?)(?:\s+\d+)?$", f, re.I)
        if m:
            v = re.sub(r"\s+\d+$", "", m.group(1).strip()).replace(" ", "_")
            return "Male", f"Male_Beard_{v}", "Male"
        return "Male", "Male_Base", "Male"
    return "Unknown", "Unknown", "Unknown"


def side_points(side):
    if side == "left":
        return list(range(FRONTAL_VIEWPOINT, MIN_VIEWPOINT - 1, -1))
    if side == "right":
        return list(range(FRONTAL_VIEWPOINT, MAX_VIEWPOINT + 1))
    raise ValueError(side)


def distance(v, side):
    if not finite(v):
        return np.nan
    v = int(v)
    if side == "left" and MIN_VIEWPOINT <= v <= FRONTAL_VIEWPOINT:
        return FRONTAL_VIEWPOINT - v
    if side == "right" and FRONTAL_VIEWPOINT <= v <= MAX_VIEWPOINT:
        return v - FRONTAL_VIEWPOINT
    return np.nan


def clean_sequence(g):
    if g.empty:
        return None
    folder = txt(g[COL_FOLDER].iloc[0])
    expr = txt(g[COL_EXPR].iloc[0])
    x = g.copy()
    x[COL_VIEW] = num(x[COL_VIEW])
    x = x[x[COL_VIEW].notna()].copy()
    x[COL_VIEW] = x[COL_VIEW].astype(int)
    x = x[x[COL_VIEW].between(MIN_VIEWPOINT, MAX_VIEWPOINT)].copy()
    if x[COL_VIEW].duplicated().any():
        return None
    if set(x[COL_VIEW]) != set(range(MIN_VIEWPOINT, MAX_VIEWPOINT + 1)):
        return None
    x = x.sort_values(COL_VIEW).reset_index(drop=True)
    x["_A"] = num(x[COL_A])
    x["_C"] = num(x[COL_C])
    x["_B"] = x[COL_B].map(txt)
    sex, appearance, family = parse_identity(folder)
    return dict(folder=folder, expression=expr, sex_identity=sex,
                appearance_identity=appearance, identity_family=family, data=x)


def build_side_event(seq, side):
    x = seq["data"]
    pts = side_points(side)
    order = {v: i for i, v in enumerate(pts)}
    y = x[x[COL_VIEW].isin(pts)].copy()
    y["_order"] = y[COL_VIEW].map(order)
    y = y.sort_values("_order").reset_index(drop=True)
    y = y[y[COL_VIEW] != FRONTAL_VIEWPOINT].reset_index(drop=True)
    if y.empty:
        return None
    views = y[COL_VIEW].to_numpy()
    a = y["_A"].to_numpy()
    c = y["_C"].to_numpy()
    b = y["_B"].to_numpy()
    ai = sustained_ge(a, A_THRESHOLD)
    ci = sustained_le(c, C_THRESHOLD)
    bi = sustained_B(b, seq["folder"])
    av = float(views[ai]) if ai is not None else np.nan
    cv = float(views[ci]) if ci is not None else np.nan
    bv = float(views[bi]) if bi is not None else np.nan
    ad = distance(av, side); cd = distance(cv, side); bd = distance(bv, side)
    lead = bd - ad if finite(ad) and finite(bd) else np.nan
    return {
        "folder": seq["folder"], "expression": seq["expression"],
        "sex_identity": seq["sex_identity"],
        "appearance_identity": seq["appearance_identity"],
        "identity_family": seq["identity_family"], "side": side,
        "A_view": av, "C_view": cv, "B_view": bv,
        "A_distance": ad, "C_distance": cd, "B_distance": bd,
        "A_B_lead": lead,
        "A_before_B": int(finite(lead) and lead > 0) if finite(lead) else 0,
        "n_viewpoints": EXPECTED_VIEWPOINTS,
        "reference_viewpoint": FRONTAL_VIEWPOINT,
        "true_folder": seq["folder"],
    }


def build_events(df):
    banner("BUILDING IDENTITY / SEQUENCE EVENTS")
    folders = sorted(df[COL_FOLDER].dropna().astype(str).unique())
    print(f"Total folders: {len(folders)}")
    events, bad = [], []
    for i, folder in enumerate(folders, 1):
        g = df[df[COL_FOLDER].astype(str) == folder]
        seq = clean_sequence(g)
        if seq is None:
            bad.append({"folder": folder, "reason": "not exactly one complete 0..214 sequence"})
        else:
            for side in ("left", "right"):
                e = build_side_event(seq, side)
                if e is not None:
                    events.append(e)
        if i <= 5 or i % 50 == 0 or i == len(folders):
            print(f"Processed {i}/{len(folders)}")
    edf = pd.DataFrame(events)
    bdf = pd.DataFrame(bad)
    print(f"\nComplete sequences: {edf['folder'].nunique() if not edf.empty else 0}")
    print(f"Side-event rows: {len(edf)}")
    print(f"Incomplete/invalid folders excluded: {len(bdf)}")
    return edf, bdf


def identity_summary(edf):
    rows=[]
    for (identity, side), g in edf.groupby(["sex_identity","side"]):
        valid=g[np.isfinite(g["A_B_lead"])].copy(); nseq=g.folder.nunique(); n=len(valid)
        if n:
            k=int(valid.A_before_B.sum()); rate=k/n; lo,hi=wilson(k,n)
            med=float(valid.A_B_lead.median()); mean=float(valid.A_B_lead.mean())
        else:
            rate=lo=hi=med=mean=np.nan
        rows.append(dict(identity=identity,side=side,n_sequences=nseq,n_valid_A_B=n,
                         A_before_B=rate,A_before_B_percent=rate*100 if finite(rate) else np.nan,
                         ci_low=lo,ci_high=hi,median_lead=med,mean_lead=mean,
                         positive_lead_percent=rate*100 if finite(rate) else np.nan))
    return pd.DataFrame(rows)


def expression_summary(edf):
    rows=[]
    for (identity,side,expr),g in edf.groupby(["sex_identity","side","expression"]):
        v=g[np.isfinite(g.A_B_lead)]; n=len(g); nv=len(v)
        if nv:
            k=int(v.A_before_B.sum()); rate=k/nv; lo,hi=wilson(k,nv); med=float(v.A_B_lead.median()); mean=float(v.A_B_lead.mean())
        else: rate=lo=hi=med=mean=np.nan
        rows.append(dict(identity=identity,side=side,expression=expr,n_sequences=n,n_valid_A_B=nv,
                         primary_eligible=n>=MIN_PRIMARY_SEQUENCES,A_before_B=rate,
                         A_before_B_percent=rate*100 if finite(rate) else np.nan,
                         ci_low=lo,ci_high=hi,median_lead=med,mean_lead=mean,
                         positive_lead_percent=rate*100 if finite(rate) else np.nan))
    return pd.DataFrame(rows)


def appearance_summary(edf):
    rows=[]
    for appearance,g in edf.groupby("appearance_identity"):
        v=g[np.isfinite(g.A_B_lead)]; nseq=g.folder.nunique(); nv=len(v)
        if nv:
            k=int(v.A_before_B.sum()); rate=k/nv; lo,hi=wilson(k,nv); med=float(v.A_B_lead.median()); mean=float(v.A_B_lead.mean())
        else: k=0; rate=lo=hi=med=mean=np.nan
        rows.append(dict(appearance_identity=appearance,n_sequences=nseq,n_valid_A_B=nv,
                         A_before_B=rate,A_before_B_percent=rate*100 if finite(rate) else np.nan,
                         ci_low=lo,ci_high=hi,median_lead=med,mean_lead=mean))
    return pd.DataFrame(rows).sort_values(["n_sequences","appearance_identity"],ascending=[False,True])


def coverage(edf):
    seq=edf[["folder","expression","sex_identity","appearance_identity"]].drop_duplicates()
    sex=seq.groupby("sex_identity").agg(n_sequences=("folder","nunique"),n_expressions=("expression","nunique")).reset_index()
    app=seq.groupby("appearance_identity").agg(n_sequences=("folder","nunique"),n_expressions=("expression","nunique")).reset_index()
    return sex,app


def horizon_summary(edf):
    rows=[]
    for identity in ["Female","Male"]:
        for side in ["left","right"]:
            v=edf[(edf.sex_identity==identity)&(edf.side==side)&np.isfinite(edf.A_B_lead)].copy(); n=len(v)
            for h in HORIZONS:
                # VALIDATED HORIZON DEFINITION: 0 < lead <= H
                k=int(((v.A_B_lead>0)&(v.A_B_lead<=h)).sum()) if n else 0
                r=k/n if n else np.nan; lo,hi=wilson(k,n) if n else (np.nan,np.nan)
                rows.append(dict(identity=identity,side=side,horizon=h,warning_count=k,n=n,
                                 warning_rate=r,warning_rate_percent=r*100 if finite(r) else np.nan,ci_low=lo,ci_high=hi))
                print(f"{identity:7s} {side:5s} Horizon={h:3d}° | warning={k}/{n} | {r*100:.3f}%" if n else f"{identity:7s} {side:5s} Horizon={h:3d}° | warning=0/0 | nan")
    return pd.DataFrame(rows)


def bootstrap_summary(edf):
    rows=[]
    for identity in ["Female","Male"]:
        for side in ["left","right"]:
            v=edf[(edf.sex_identity==identity)&(edf.side==side)].A_before_B.to_numpy(dtype=float)
            seed=SEED+(0 if identity=="Female" else 100)+(0 if side=="left" else 10)
            m,lo,hi=bootstrap_mean(v,BOOTSTRAP_REPS,seed)
            rows.append(dict(identity=identity,side=side,n=len(v),observed_rate=m,ci_low=lo,ci_high=hi,bootstrap_reps=BOOTSTRAP_REPS))
            print(f"{identity:7s} {side:5s} mean={m:.4f} CI=[{lo:.4f}, {hi:.4f}]")
    return pd.DataFrame(rows)


def heterogeneity_summary(edf):
    rows=[]
    for identity in ["Female","Male"]:
        for side in ["left","right"]:
            g=edf[(edf.sex_identity==identity)&(edf.side==side)]
            groups=[x.A_before_B.to_numpy(float) for _,x in g.groupby("expression")]
            obs,nm,p=range_perm(groups,PERMUTATION_REPS,SEED+(1 if identity=="Male" else 0)+(10 if side=="right" else 0))
            rows.append(dict(identity=identity,side=side,n_expression_groups=len(groups),observed_range=obs,null_mean=nm,p_value=p))
            print(f"{identity:7s} {side:5s} observed range={obs:.4f} null mean={nm:.4f} p={p:.6f}" if finite(obs) else f"{identity:7s} {side:5s} observed range=nan null mean=nan p=nan")
    return pd.DataFrame(rows)


def loo_summary(edf):
    rows=[]
    exprs=sorted(edf.expression.dropna().unique())
    for ex in exprs:
        row={"excluded_expression":ex}
        for identity in ["Female","Male"]:
            for side in ["left","right"]:
                v=edf[(edf.sex_identity==identity)&(edf.side==side)&(edf.expression!=ex)]
                v=v[np.isfinite(v.A_B_lead)]
                p=f"{identity}_{side}"
                row[f"{p}_A_before_B"]=float(v.A_before_B.mean()) if len(v) else np.nan
                row[f"{p}_median_lead"]=float(v.A_B_lead.median()) if len(v) else np.nan
                row[f"{p}_mean_lead"]=float(v.A_B_lead.mean()) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def robustness(edf):
    rows=[]
    for identity in ["Female","Male"]:
        for side in ["left","right"]:
            v=edf[(edf.sex_identity==identity)&(edf.side==side)]
            v=v[np.isfinite(v.A_B_lead)]
            if v.empty: continue
            base=float(v.A_before_B.mean()); med=float(v.A_B_lead.median())
            rates=v.groupby("expression").A_before_B.mean(); meds=v.groupby("expression").A_B_lead.median()
            f80=float((rates>=.8).mean()); f70=float((rates>=.7).mean()); fp=float((meds>0).mean())
            score=(base+f80+f70+fp)/4
            rows.append(dict(identity=identity,side=side,baseline_A_before_B=base,median_lead=med,
                             fraction_expression_A_before_B_ge_80=f80,fraction_expression_A_before_B_ge_70=f70,
                             fraction_expression_positive_median_lead=fp,overall_identity_robustness_score=score))
    return pd.DataFrame(rows)


def paired_comparison(edf):
    # Pair at expression + side level. Multiple raw folders within an expression are aggregated first.
    def agg(identity):
        x=edf[edf.sex_identity==identity]
        rows=[]
        for (expr,side),g in x.groupby(["expression","side"]):
            v=g[np.isfinite(g.A_B_lead)]
            if v.empty: continue
            rows.append(dict(expression=expr,side=side,A_before_B=float(v.A_before_B.mean()),
                             mean_lead=float(v.A_B_lead.mean()),median_lead=float(v.A_B_lead.median()),n_sequences=v.folder.nunique()))
        return pd.DataFrame(rows)
    f=agg("Female"); m=agg("Male")
    p=f.merge(m,on=["expression","side"],suffixes=("_female","_male"),how="inner")
    if p.empty: return p,np.nan,np.nan,np.nan,np.nan
    rdiff,rp=paired_perm(p.A_before_B_female,p.A_before_B_male,PERMUTATION_REPS,SEED+500)
    ldiff,lp=paired_perm(p.mean_lead_female,p.mean_lead_male,PERMUTATION_REPS,SEED+501)
    p["rate_difference_female_minus_male"]=p.A_before_B_female-p.A_before_B_male
    p["mean_lead_difference_female_minus_male"]=p.mean_lead_female-p.mean_lead_male
    return p,rdiff,rp,ldiff,lp


def save_csv(df,name):
    p=OUTPUT_DIR/name
    df.to_csv(p,index=False,encoding="utf-8-sig")
    print(p)
    return p


def plot_ab(summary,path):
    if summary.empty: return
    x=np.arange(len(summary)); y=summary.A_before_B_percent.fillna(0).to_numpy()
    fig,ax=plt.subplots(figsize=(9,6)); ax.bar(x,y); ax.set_xticks(x); ax.set_xticklabels([f"{r.identity} / {r.side}" for r in summary.itertuples()],rotation=25,ha="right"); ax.set_ylabel("A-before-B (%)"); ax.set_ylim(0,100); ax.set_title("Cross-Identity A-before-B Rate"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)


def plot_horizon(h,path):
    if h.empty: return
    fig,ax=plt.subplots(figsize=(10,6))
    for (identity,side),g in h.groupby(["identity","side"]):
        g=g.sort_values("horizon"); ax.plot(g.horizon,g.warning_rate_percent,marker="o",label=f"{identity} / {side}")
    ax.set_xlabel("Horizon (degrees)"); ax.set_ylabel("Warning rate (%)"); ax.set_ylim(0,100); ax.set_title("Cross-Identity Horizon Analysis"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)


def plot_lead(edf,path):
    v=edf[np.isfinite(edf.A_B_lead)]
    if v.empty:return
    groups=[]; labels=[]
    for (identity,side),g in v.groupby(["sex_identity","side"]): groups.append(g.A_B_lead.to_numpy()); labels.append(f"{identity} / {side}")
    fig,ax=plt.subplots(figsize=(10,6)); ax.boxplot(groups,labels=labels,showmeans=True); ax.axhline(0,linewidth=1); ax.set_ylabel("Lead = B distance - A distance (degrees)"); ax.set_title("Identity Lead Distribution"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)


def plot_lr(summary,path):
    p=summary.pivot(index="identity",columns="side",values="A_before_B_percent"); fig,ax=plt.subplots(figsize=(8,6)); x=np.arange(len(p)); w=.35
    if "left" in p: ax.bar(x-w/2,p.left.fillna(0),w,label="left")
    if "right" in p: ax.bar(x+w/2,p.right.fillna(0),w,label="right")
    ax.set_xticks(x); ax.set_xticklabels(p.index); ax.set_ylabel("A-before-B (%)"); ax.set_ylim(0,100); ax.set_title("Identity Left / Right Validation"); ax.grid(axis="y",alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)


def records(df):
    if df is None or df.empty:return []
    out=[]
    for r in df.to_dict("records"):
        q={}
        for k,v in r.items():
            if isinstance(v,(np.integer,)): v=int(v)
            elif isinstance(v,(np.floating,)): v=None if not np.isfinite(v) else float(v)
            q[k]=v
        out.append(q)
    return out


def write_readme():
    text=f'''CROSS-IDENTITY ANALYSIS\n=======================\n\nA threshold: {A_THRESHOLD:.8f}\nC threshold: {C_THRESHOLD:.8f}\nSustained viewpoints: {SUSTAINED_VIEWPOINTS}\nFrontal viewpoint: {FRONTAL_VIEWPOINT}\n\nA: A_angular_distance_deg >= threshold for 3 consecutive viewpoints.\nC: C_margin <= threshold for 3 consecutive viewpoints.\nB: B_predicted_folder != TRUE FOLDER for 3 consecutive viewpoints.\n\nTraversal:\n  left  = 107 -> 106 -> ... -> 0\n  right = 107 -> 108 -> ... -> 214\n\nLead:\n  B_distance_from_frontal - A_distance_from_frontal\n  lead > 0 means A-before-B.\n\nA horizon definition:\n  0 < lead <= H\n\nSequences must contain exactly one row for every viewpoint 0..214.\nFemale/Male pairing is expression + side level, never a raw sequence Cartesian product.\nSmall appearance groups are descriptive only.\nPositive A-before-B is precedence, not causality.\n'''
    (OUTPUT_DIR/"README_cross_identity.md").write_text(text,encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    banner("CROSS-IDENTITY ANALYSIS\nVALIDATED / CONSISTENT PIPELINE")
    print("Project root:"); print(PROJECT_ROOT); print("\nInput:"); print(INPUT_FILE); print("\nOutput:"); print(OUTPUT_DIR)
    banner("LOADING DATA")
    if not INPUT_FILE.exists(): raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")
    df=pd.read_csv(INPUT_FILE,low_memory=False)
    print(f"Rows loaded: {len(df):,}\nColumns found: {len(df.columns)}")
    required=[COL_EXPR,COL_FOLDER,COL_VIEW,COL_A,COL_C,COL_B]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError("Missing required columns:\n"+"\n".join(missing))
    banner("COLUMN MAPPING")
    print(f"expression      : {COL_EXPR}\nfolder          : {COL_FOLDER}\nviewpoint       : {COL_VIEW}\nA               : {COL_A}\nC               : {COL_C}\nB               : {COL_B}")
    banner("VALIDATED THRESHOLDS — FIXED")
    print(f"A threshold : {A_THRESHOLD:.8f}\nC threshold : {C_THRESHOLD:.8f}\nSustained viewpoints : {SUSTAINED_VIEWPOINTS}")
    print("\nNO THRESHOLD ESTIMATION IS PERFORMED IN THIS SCRIPT.")
    print("\nB EVENT:\npredicted_folder != true folder\nfor 3 consecutive viewpoints.")
    print("\nLEFT TRAVERSAL:\n107 -> 106 -> ... -> 0\nRIGHT TRAVERSAL:\n107 -> 108 -> ... -> 214")
    df[COL_VIEW]=num(df[COL_VIEW]); df[COL_A]=num(df[COL_A]); df[COL_C]=num(df[COL_C]); df[COL_FOLDER]=df[COL_FOLDER].map(txt); df[COL_EXPR]=df[COL_EXPR].map(txt); df[COL_B]=df[COL_B].map(txt)
    df=df[(df[COL_FOLDER]!="")&(df[COL_EXPR]!="")&df[COL_VIEW].notna()].copy()
    edf,bad=build_events(df)
    if edf.empty: raise RuntimeError("No complete sequences were found.")

    banner("IDENTITY COVERAGE")
    sex_cov,app_cov=coverage(edf); print(sex_cov.to_string(index=False)); print(); print(app_cov.to_string(index=False))

    banner("PRIMARY IDENTITY ELIGIBILITY")
    expr_cov=(edf[["folder","expression","sex_identity"]].drop_duplicates().groupby(["sex_identity","expression"]).agg(n_sequences=("folder","nunique")).reset_index())
    expr_cov["primary_eligible"]=expr_cov.n_sequences>=MIN_PRIMARY_SEQUENCES
    print(expr_cov.to_string(index=False))

    banner("SEX / IDENTITY SUMMARY")
    ids=identity_summary(edf); print(ids.to_string(index=False))

    banner("EXPRESSION-LEVEL IDENTITY ANALYSIS")
    expr=expression_summary(edf); print(expr.to_string(index=False))

    banner("APPEARANCE IDENTITY ANALYSIS")
    app=appearance_summary(edf); print(app.to_string(index=False))

    banner("PAIRED FEMALE / MALE IDENTITY ANALYSIS")
    paired,rate_diff,rate_p,lead_diff,lead_p=paired_comparison(edf)
    print(f"Paired expression-side observations: {len(paired)}")
    print(f"Female - Male A-before-B difference={rate_diff:.6f}")
    print(f"Permutation p-value={rate_p:.6f}")
    print(f"Female - Male mean lead difference={lead_diff:.6f}°")
    print(f"Permutation p-value={lead_p:.6f}")

    banner("CROSS-IDENTITY HORIZON ANALYSIS")
    hor=horizon_summary(edf)

    banner("BOOTSTRAP STABILITY")
    boot=bootstrap_summary(edf)

    banner("EXPRESSION-LEVEL IDENTITY HETEROGENEITY")
    het=heterogeneity_summary(edf)

    banner("LEAVE-ONE-EXPRESSION-OUT")
    loo=loo_summary(edf)

    banner("IDENTITY ROBUSTNESS SCORE")
    rob=robustness(edf); print(rob.to_string(index=False))

    banner("FINAL CROSS-IDENTITY SUMMARY")
    print(f"Sequences analyzed: {edf.folder.nunique()}")
    print(f"Expressions analyzed: {edf.expression.nunique()}")
    for r in ids.itertuples(): print(f"{r.identity:7s} {r.side:5s} A-before-B={r.A_before_B_percent:.3f}% median lead={r.median_lead:.3f}°")
    print("\nPRIMARY IDENTITY COMPARISON")
    print(f"Female - Male A-before-B difference={rate_diff:.6f}")
    print(f"Permutation p-value={rate_p:.6f}")
    print(f"Female - Male mean lead difference={lead_diff:.6f}°")
    print(f"Permutation p-value={lead_p:.6f}")
    print("\nIMPORTANT:")
    print("A/C thresholds are fixed; no threshold re-estimation is performed.")
    print("B uses predicted_folder != true folder with the sustained 3-view rule.")
    print("Left/right traversal is outward from frontal viewpoint 107.")
    print("Female/Male pairing is expression + side level, not raw sequence Cartesian products.")
    print("Small appearance groups are descriptive only.")
    print("Positive A-before-B indicates temporal/statistical precedence, not causality.")

    banner("SAVING RESULTS")
    save_csv(edf,"identity_side_sequence_events.csv")
    save_csv(edf,"identity_sequence_events.csv")
    save_csv(sex_cov,"identity_coverage.csv")
    save_csv(app_cov,"identity_appearance_coverage.csv")
    save_csv(expr_cov,"identity_expression_coverage.csv")
    save_csv(ids,"identity_side_summary.csv")
    save_csv(ids,"identity_summary.csv")
    save_csv(expr,"identity_side_expression_summary.csv")
    save_csv(expr,"identity_expression_summary.csv")
    save_csv(app,"appearance_identity_summary.csv")
    save_csv(paired,"paired_identity_comparison.csv")
    save_csv(hor,"identity_horizon_summary.csv")
    save_csv(boot,"identity_bootstrap.csv")
    save_csv(het,"identity_heterogeneity.csv")
    save_csv(loo,"identity_leave_one_expression_out.csv")
    save_csv(rob,"identity_robustness_score.csv")
    save_csv(bad,"identity_incomplete_sequences.csv")

    banner("BUILDING PLOTS")
    plot_ab(ids,OUTPUT_DIR/"identity_A_before_B.png")
    plot_horizon(hor,OUTPUT_DIR/"identity_horizon.png")
    plot_lead(edf,OUTPUT_DIR/"identity_lead_distribution.png")
    plot_lr(ids,OUTPUT_DIR/"identity_left_right.png")
    for p in ["identity_A_before_B.png","identity_horizon.png","identity_lead_distribution.png","identity_left_right.png"]: print(OUTPUT_DIR/p)

    write_readme(); print(OUTPUT_DIR/"README_cross_identity.md")
    report={"project_root":str(PROJECT_ROOT),"input_file":str(INPUT_FILE),"output_dir":str(OUTPUT_DIR),"configuration":{"A_threshold":A_THRESHOLD,"C_threshold":C_THRESHOLD,"sustained_viewpoints":SUSTAINED_VIEWPOINTS,"frontal_viewpoint":FRONTAL_VIEWPOINT,"expected_viewpoints":EXPECTED_VIEWPOINTS,"horizons":HORIZONS,"min_primary_sequences":MIN_PRIMARY_SEQUENCES},"event_definitions":{"A":"A_angular_distance_deg >= A_THRESHOLD for 3 consecutive viewpoints","C":"C_margin <= C_THRESHOLD for 3 consecutive viewpoints","B":"B_predicted_folder != true folder for 3 consecutive viewpoints"},"traversal":{"left":"107 -> 106 -> ... -> 0","right":"107 -> 108 -> ... -> 214"},"horizon_definition":"0 < lead <= H","lead_definition":"B_distance_from_frontal - A_distance_from_frontal","n_complete_sequences":int(edf.folder.nunique()),"n_expressions":int(edf.expression.nunique()),"identity_summary":records(ids),"expression_summary":records(expr),"appearance_summary":records(app),"paired_comparison":records(paired),"horizon_summary":records(hor),"bootstrap":records(boot),"heterogeneity":records(het),"robustness":records(rob),"important_notes":["A/C thresholds are fixed.","B compares prediction against the true folder.","Left/right traversal starts at frontal viewpoint 107.","Female/Male pairing is expression + side level.","Small appearance groups are descriptive only.","Positive A-before-B indicates precedence, not causality."]}
    # JSON does not permit NaN/Infinity. Non-estimable statistics (for example
    # incomplete one-sequence appearance groups) are written as null.
    report = json_safe(report)
    with open(OUTPUT_DIR/"cross_identity_report.json","w",encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(OUTPUT_DIR/"cross_identity_report.json")
    banner("DONE")


if __name__ == "__main__":
    main()
