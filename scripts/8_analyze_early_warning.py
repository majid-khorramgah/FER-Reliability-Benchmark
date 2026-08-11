from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, precision_recall_curve
)

# ============================================================
# EARLY-WARNING PREDICTION / STATISTICAL VALIDATION
# ============================================================
# Input:
#   analysis/per_view_metrics_multimetric.csv
#
# Main scientific question:
#   Does representation drift A provide useful early warning
#   of later prediction failure B?
#
# This script tests:
# 1. A-before-B boundary precedence, separately LEFT/RIGHT
# 2. Permutation null for A-before-B
# 3. Viewpoint-only vs A vs A+C predictive models
# 4. Grouped cross-validation by expression/folder
# 5. Bootstrap confidence intervals for model AUC
# 6. Calibration
# 7. ROC/PR curves
# 8. Effect of A after controlling for viewpoint
# 9. Gender subgroup robustness (from folder prefix)
# 10. Threshold sensitivity
# 11. Early-warning horizon analysis
#
# IMPORTANT:
# Significant prediction/precedence is NOT causality.
# ============================================================

ROOT = Path(__file__).resolve().parents[0]
ANALYSIS = ROOT / "analysis"
INPUT = ANALYSIS / "4_analyze_embeddings_trajectory" / "per_view_metrics_multimetric.csv"
OUT = ANALYSIS / "8_analyze_early_warning"
PLOTS = OUT / "plots"

FRONTAL = 107
EXPECTED_VIEWS = 215
DEFAULT_A_THRESHOLD = 13.43702602
DEFAULT_C_THRESHOLD = 0.00237080
SUSTAINED = 3
N_BOOT = 1000
N_PERM = 2000
SEED = 1405

rng = np.random.default_rng(SEED)


def find_col(df, names, required=True):
    exact = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in exact:
            return exact[n.lower()]
    norm = {
        "".join(x for x in str(c).lower() if x.isalnum()): c
        for c in df.columns
    }
    for n in names:
        k = "".join(x for x in n.lower() if x.isalnum())
        if k in norm:
            return norm[k]
    if required:
        raise RuntimeError(
            f"Missing required column. Tried {names}. "
            f"Available: {list(df.columns)}"
        )
    return None


def columns(df):
    return {
        "expression": find_col(df, ["expression"]),
        "folder": find_col(df, ["folder"]),
        "viewpoint": find_col(df, ["viewpoint", "angle"]),
        "A": find_col(df, [
            "A_angular_distance_deg",
            "A_angular_distance",
            "A_angular"
        ]),
        "C": find_col(df, ["C_margin"]),
        "B": find_col(df, [
            "B_predicted_folder",
            "B_predicted",
            "B"
        ]),
        "rate": find_col(df, [
            "A_rate_per_degree", "A_rate"
        ], False),
        "curvature": find_col(df, ["A_curvature"], False),
        "instability": find_col(df, [
            "A_trajectory_instability", "A_instability"
        ], False),
        "path": find_col(df, [
            "A_cumulative_path_from_V107", "A_path"
        ], False),
    }


def load_thresholds():
    t = {"A": DEFAULT_A_THRESHOLD, "C": DEFAULT_C_THRESHOLD}
    candidates = list(ANALYSIS.rglob("*report.json"))
    for p in candidates:
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield str(k).lower(), v
                    yield from walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from walk(v)

        for k, v in walk(x):
            if not isinstance(v, (int, float)):
                continue
            if k in {
                "a_angular_threshold", "a_threshold",
                "a_drift_threshold"
            } and np.isfinite(v):
                t["A"] = float(v)
            if k in {"c_margin_threshold", "c_threshold"} and np.isfinite(v):
                t["C"] = float(v)
    return t


def clean_num(s):
    return pd.to_numeric(s, errors="coerce")


def first_sustained(v, cond, side):
    v = np.asarray(v, float)
    cond = np.asarray(cond, bool)
    m = np.isfinite(v)
    v, cond = v[m], cond[m]
    if side == "left":
        m = v <= FRONTAL
        v, cond = v[m], cond[m]
        order = np.argsort(v)[::-1]
    else:
        m = v >= FRONTAL
        v, cond = v[m], cond[m]
        order = np.argsort(v)
    v, cond = v[order], cond[order]
    run = 0
    for i, ok in enumerate(cond):
        run = run + 1 if ok else 0
        if run >= SUSTAINED:
            return float(v[i - SUSTAINED + 1])
    return np.nan


def make_events(df, c, thresholds):
    rows = []
    groups = list(df.groupby(c["folder"], sort=False))
    for i, (folder, g) in enumerate(groups, 1):
        vp = clean_num(g[c["viewpoint"]]).dropna().astype(int)
        if len(vp) != EXPECTED_VIEWS or set(vp) != set(range(EXPECTED_VIEWS)):
            continue

        v = clean_num(g[c["viewpoint"]]).to_numpy()
        A = clean_num(g[c["A"]]).to_numpy()
        C = clean_num(g[c["C"]]).to_numpy()
        pred = g[c["B"]].astype(str).str.strip()
        truth = str(folder)
        valid_pred = ~pred.str.lower().isin(
            {"", "nan", "none", "null", "na", "n/a"}
        )
        Bfail = (valid_pred & (pred != truth)).to_numpy()

        row = {"expression": str(g[c["expression"]].iloc[0]),
               "folder": str(folder)}
        for side in ("left", "right"):
            row[f"A_{side}"] = first_sustained(
                v, np.isfinite(A) & (A >= thresholds["A"]), side)
            row[f"C_{side}"] = first_sustained(
                v, np.isfinite(C) & (C <= thresholds["C"]), side)
            row[f"B_{side}"] = first_sustained(v, Bfail, side)
        rows.append(row)
        if i <= 5 or i % 50 == 0 or i == len(groups):
            print(f"Processed {i}/{len(groups)}")
    return pd.DataFrame(rows)


def make_prediction_table(df, c, thresholds):
    x = pd.DataFrame()
    x["expression"] = df[c["expression"]].astype(str)
    x["folder"] = df[c["folder"]].astype(str)
    x["viewpoint"] = clean_num(df[c["viewpoint"]])
    x["A"] = clean_num(df[c["A"]])
    x["C"] = clean_num(df[c["C"]])

    pred = df[c["B"]].astype(str).str.strip()
    valid = ~pred.str.lower().isin({"", "nan", "none", "null", "na", "n/a"})
    x["B_failure"] = (valid & (pred != x["folder"])).astype(int)

    x["side"] = np.where(
        x.viewpoint < FRONTAL, "left",
        np.where(x.viewpoint > FRONTAL, "right", "frontal")
    )
    x["distance"] = (x.viewpoint - FRONTAL).abs()
    x["A_excess"] = (x.A - thresholds["A"]).clip(lower=0)
    x["C_deficit"] = (thresholds["C"] - x.C).clip(lower=0)

    for new, src in [
        ("A_rate", c["rate"]),
        ("A_curvature", c["curvature"]),
        ("A_instability", c["instability"]),
        ("A_path", c["path"]),
    ]:
        x[new] = clean_num(df[src]) if src else np.nan

    return x


def pipe(features):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=3000, class_weight="balanced",
            solver="liblinear", random_state=SEED
        ))
    ])


def grouped_cv(data, features, n_splits=5, return_scores=False):
    d = data[features + ["B_failure", "folder"]].replace(
        [np.inf, -np.inf], np.nan
    )
    y = d.B_failure.astype(int).to_numpy()
    groups = d.folder.astype(str).to_numpy()

    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return {"n": len(y), "positive": int(y.sum()),
                "auc": np.nan, "ap": np.nan, "brier": np.nan,
                "accuracy": np.nan, "balanced_accuracy": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan}

    k = min(n_splits, len(np.unique(groups)))
    splitter = GroupKFold(k)
    ys, ps, preds = [], [], []

    for tr, te in splitter.split(d[features], y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        model = pipe(features)
        model.fit(d[features].iloc[tr], y[tr])
        p = model.predict_proba(d[features].iloc[te])[:, 1]
        ys.extend(y[te]); ps.extend(p); preds.extend((p >= .5).astype(int))

    ys, ps, preds = np.array(ys), np.array(ps), np.array(preds)
    if len(np.unique(ys)) < 2:
        auc = ap = np.nan
    else:
        auc = roc_auc_score(ys, ps)
        ap = average_precision_score(ys, ps)

    result = {
        "n": len(ys),
        "positive": int(ys.sum()),
        "negative": int(len(ys) - ys.sum()),
        "auc": float(auc),
        "ap": float(ap),
        "brier": float(brier_score_loss(ys, ps)),
        "accuracy": float(accuracy_score(ys, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(ys, preds)),
        "precision": float(precision_score(ys, preds, zero_division=0)),
        "recall": float(recall_score(ys, preds, zero_division=0)),
        "f1": float(f1_score(ys, preds, zero_division=0)),
    }
    if return_scores:
        result["_y"], result["_p"] = ys, ps
    return result


def model_comparison(data):
    specs = {
        "viewpoint_only": ["distance"],
        "A_only": ["A"],
        "C_only": ["C"],
        "A_plus_viewpoint": ["A", "distance"],
        "C_plus_viewpoint": ["C", "distance"],
        "A_plus_C": ["A", "C"],
        "A_plus_C_plus_viewpoint": ["A", "C", "distance"],
        "A_trajectory": [
            "A", "A_rate", "A_curvature",
            "A_instability", "A_path", "distance"
        ],
    }
    rows = []
    scores = {}
    for name, f in specs.items():
        print("Grouped CV:", name)
        r = grouped_cv(data, f, return_scores=True)
        scores[name] = (r.pop("_y", None), r.pop("_p", None))
        r["model"] = name
        r["features"] = ",".join(f)
        rows.append(r)
    return pd.DataFrame(rows), scores, specs


def bootstrap_auc(data, features, n=N_BOOT):
    groups = data.folder.astype(str).unique()
    vals = []
    for _ in range(n):
        chosen = rng.choice(groups, len(groups), replace=True)
        b = pd.concat(
            [data[data.folder.astype(str) == g] for g in chosen],
            ignore_index=True
        )
        y = b.B_failure.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        m = pipe(features)
        try:
            m.fit(b[features], y)
            p = m.predict_proba(b[features])[:, 1]
            vals.append(roc_auc_score(y, p))
        except Exception:
            pass
    if not vals:
        return np.nan, np.nan, np.nan
    return np.median(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def permutation_ordering(events, side, n=N_PERM):
    A = np.abs(FRONTAL - events[f"A_{side}"].to_numpy(float))
    B = np.abs(FRONTAL - events[f"B_{side}"].to_numpy(float))
    ok = np.isfinite(A) & np.isfinite(B)
    A, B = A[ok], B[ok]
    if not len(A):
        return np.nan, np.nan, np.nan, np.nan
    observed = np.mean(A < B)
    null = np.empty(n)
    for i in range(n):
        null[i] = np.mean(A < rng.permutation(B))
    p = (1 + np.sum(null >= observed)) / (n + 1)
    return observed, np.mean(null), np.percentile(null, 2.5), p


def boundary_summary(events):
    rows = []
    for side in ("left", "right"):
        A = np.abs(FRONTAL - events[f"A_{side}"])
        C = np.abs(FRONTAL - events[f"C_{side}"])
        B = np.abs(FRONTAL - events[f"B_{side}"])
        ok = np.isfinite(A) & np.isfinite(C) & np.isfinite(B)
        A, C, B = A[ok], C[ok], B[ok]
        rows.append({
            "side": side, "n": len(A),
            "A_before_B": np.mean(A < B) if len(A) else np.nan,
            "A_before_C": np.mean(A < C) if len(A) else np.nan,
            "C_before_B": np.mean(C < B) if len(A) else np.nan,
            "median_A_distance": np.median(A) if len(A) else np.nan,
            "median_C_distance": np.median(C) if len(A) else np.nan,
            "median_B_distance": np.median(B) if len(A) else np.nan,
            "median_A_B_lead": np.median(B - A) if len(A) else np.nan,
        })
    return pd.DataFrame(rows)


def horizon_analysis(events, horizons=(5, 10, 20, 30, 40, 60)):
    rows = []
    for side in ("left", "right"):
        A = np.abs(FRONTAL - events[f"A_{side}"].to_numpy(float))
        B = np.abs(FRONTAL - events[f"B_{side}"].to_numpy(float))
        ok = np.isfinite(A) & np.isfinite(B)
        A, B = A[ok], B[ok]
        for h in horizons:
            # A must happen before B and the remaining distance must
            # be no more than h degrees.
            lead = B - A
            rows.append({
                "side": side,
                "horizon_degrees": h,
                "n": len(A),
                "A_before_B": np.mean(A < B) if len(A) else np.nan,
                "A_early_warning_within_horizon":
                    np.mean((lead > 0) & (lead <= h)) if len(A) else np.nan,
            })
    return pd.DataFrame(rows)


def gender_subgroups(data):
    x = data.copy()
    x["gender"] = np.where(
        x.folder.str.lower().str.startswith("male"), "male",
        np.where(x.folder.str.lower().str.startswith("female"),
                 "female", "unknown")
    )
    rows = []
    for g, d in x.groupby("gender"):
        if g == "unknown":
            continue
        for name, f in {
            "viewpoint_only": ["distance"],
            "A_plus_C_plus_viewpoint": ["A", "C", "distance"],
        }.items():
            r = grouped_cv(d, f)
            r["gender"] = g
            r["model"] = name
            rows.append(r)
    return pd.DataFrame(rows)


def save_roc(scores, out):
    plt.figure(figsize=(7, 7))
    for name, (y, p) in scores.items():
        if y is None or len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y,p):.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Grouped-CV ROC curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_pr(scores, out):
    plt.figure(figsize=(7, 7))
    for name, (y, p) in scores.items():
        if y is None or len(np.unique(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, p)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Grouped-CV Precision-Recall curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_model_bar(models, out):
    plt.figure(figsize=(10,6))
    plt.bar(models.model, models.auc)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Grouped-CV ROC-AUC")
    plt.ylim(0,1)
    plt.title("Early-warning model comparison")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_boundary_plot(events, out):
    plt.figure(figsize=(8,7))
    for side, marker in [("left","o"),("right","s")]:
        A = np.abs(FRONTAL-events[f"A_{side}"].to_numpy(float))
        B = np.abs(FRONTAL-events[f"B_{side}"].to_numpy(float))
        ok = np.isfinite(A)&np.isfinite(B)
        plt.scatter(A[ok], B[ok], alpha=.35, s=20, marker=marker, label=side)
    lim = [0, max(30, np.nanmax([
        np.abs(FRONTAL-events["A_left"]).max(),
        np.abs(FRONTAL-events["B_left"]).max(),
        np.abs(FRONTAL-events["A_right"]).max(),
        np.abs(FRONTAL-events["B_right"]).max()
    ]) + 3)]
    plt.plot(lim, lim, "--")
    plt.xlabel("A boundary distance from frontal (degrees)")
    plt.ylabel("B boundary distance from frontal (degrees)")
    plt.title("A vs B boundary precedence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    warnings.filterwarnings("ignore", message=".*labels.*boxplot.*")
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*70)
    print("EARLY-WARNING PREDICTION / STATISTICAL VALIDATION")
    print("#"*70)
    print("Project root:", ROOT)
    print("Input:", INPUT)

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)
    c = columns(df)
    t = load_thresholds()

    print(f"Rows loaded: {len(df):,}")
    print("\nCOLUMN MAPPING")
    for k,v in c.items():
        print(f"{k:16s}: {v}")
    print("\nTHRESHOLDS")
    print(f"A angular threshold : {t['A']:.8f}")
    print(f"C margin threshold  : {t['C']:.8f}")
    print(f"Sustained viewpoints: {SUSTAINED}")

    print("\nBUILDING SEQUENCE EVENTS")
    events = make_events(df, c, t)
    if events.empty:
        raise RuntimeError("No complete sequences found.")
    events.to_csv(OUT/"sequence_events.csv", index=False)
    print("Complete sequences:", len(events))

    # ---------------- boundary precedence ----------------
    bs = boundary_summary(events)
    bs.to_csv(OUT/"boundary_summary.csv", index=False)

    print("\nBOUNDARY PRECEDENCE")
    print(bs.to_string(index=False))

    perm_rows = []
    for side in ("left","right"):
        print(f"Permutation {side}: {N_PERM}")
        obs, nullmean, nullp, p = permutation_ordering(events, side)
        perm_rows.append({
            "side": side,
            "observed_A_before_B": obs,
            "null_mean": nullmean,
            "null_2_5": nullp,
            "p_permutation": p
        })
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(OUT/"permutation_ordering.csv", index=False)

    # ---------------- image-level prediction ----------------
    pred = make_prediction_table(df, c, t)
    pred = pred[pred.side.isin(["left","right"])].copy()
    pred.to_csv(OUT/"prediction_dataset.csv", index=False)

    print("\nB FAILURE PREVALENCE")
    print(pred.B_failure.value_counts(dropna=False).to_string())

    models, scores, specs = model_comparison(pred)
    models.to_csv(OUT/"model_comparison.csv", index=False)

    print("\nMODEL COMPARISON")
    print(models[[
        "model","n","positive","auc","ap","brier",
        "balanced_accuracy","precision","recall","f1"
    ]].to_string(index=False))

    # ---------------- bootstrap AUC ----------------
    bootrows = []
    for name, f in specs.items():
        print("Bootstrap AUC:", name)
        med, lo, hi = bootstrap_auc(pred, f)
        bootrows.append({
            "model": name,
            "median_auc": med,
            "ci_2_5": lo,
            "ci_97_5": hi,
            "n_bootstrap": N_BOOT
        })
    boot = pd.DataFrame(bootrows)
    boot.to_csv(OUT/"bootstrap_auc_ci.csv", index=False)

    # ---------------- horizon ----------------
    hz = horizon_analysis(events)
    hz.to_csv(OUT/"early_warning_horizons.csv", index=False)

    # ---------------- subgroup ----------------
    sub = gender_subgroups(pred)
    sub.to_csv(OUT/"subgroup_results.csv", index=False)

    # ---------------- coefficients ----------------
    coef_rows = []
    for name, f in {
        "A_plus_C_plus_viewpoint":["A","C","distance"],
        "A_trajectory":["A","A_rate","A_curvature","A_instability","A_path","distance"]
    }.items():
        d = pred[f+["B_failure"]].replace([np.inf,-np.inf],np.nan)
        if d.B_failure.nunique() < 2:
            continue
        m = pipe(f)
        m.fit(d[f], d.B_failure.astype(int))
        co = m.named_steps["logreg"].coef_[0]
        for feature, val in zip(f,co):
            coef_rows.append({
                "model":name,
                "feature":feature,
                "standardized_log_odds":val,
                "odds_ratio_per_1sd":np.exp(val)
            })
    pd.DataFrame(coef_rows).to_csv(
        OUT/"model_coefficients.csv", index=False
    )

    # ---------------- threshold sensitivity ----------------
    sensitivity = []
    Aall = clean_num(df[c["A"]])
    Call = clean_num(df[c["C"]])
    for q in [0.80,0.85,0.90,0.925,0.95,0.975]:
        ath = float(Aall.quantile(q))
        for cq in [0.025,0.05,0.075,0.10,0.15,0.20]:
            cth = float(Call.quantile(cq))
            sensitivity.append({
                "A_quantile":q, "A_threshold":ath,
                "C_quantile":cq, "C_threshold":cth
            })
    pd.DataFrame(sensitivity).to_csv(
        OUT/"threshold_sensitivity_grid.csv", index=False
    )

    # ---------------- plots ----------------
    save_roc(scores, PLOTS/"roc_grouped_cv.png")
    save_pr(scores, PLOTS/"pr_grouped_cv.png")
    save_model_bar(models, PLOTS/"model_auc_comparison.png")
    save_boundary_plot(events, PLOTS/"A_B_boundary_precedence.png")

    # ---------------- report ----------------
    base = models.loc[models.model=="viewpoint_only","auc"]
    primary = models.loc[
        models.model=="A_plus_C_plus_viewpoint","auc"
    ]
    gain = float(primary.iloc[0]-base.iloc[0]) if len(base) and len(primary) else np.nan

    report = {
        "project":"FER-Reliability-Benchmark",
        "question":"Does representation drift A provide early-warning information about later prediction failure B?",
        "input_rows":len(df),
        "complete_sequences":len(events),
        "thresholds":t,
        "boundary_precedence":bs.to_dict("records"),
        "permutation_ordering":perm.to_dict("records"),
        "models":models.to_dict("records"),
        "bootstrap_auc":boot.to_dict("records"),
        "horizons":hz.to_dict("records"),
        "subgroups":sub.to_dict("records"),
        "A_model_AUC_gain_over_viewpoint_baseline":gain,
        "interpretation":{
            "strong_support_for_early_warning":
                "A-before-B is high and permutation p is small AND A-based grouped-CV performance exceeds viewpoint-only baseline.",
            "important_control":
                "Viewpoint-only baseline must be beaten; otherwise A may simply be rediscovering viewpoint.",
            "causality":
                "These analyses establish temporal/statistical precedence or predictive utility, not causal direction."
        }
    }

    (OUT/"report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    readme = f"""# Early-Warning Prediction

Main question:

> Does representation drift (A) provide useful early-warning information
> about later prediction failure (B)?

Data:
- {len(df):,} image rows
- {len(events)} complete expression sequences
- 215 viewpoints per complete sequence
- frontal viewpoint = {FRONTAL}°

Thresholds:
- A = {t["A"]:.8f} degrees
- C = {t["C"]:.8f}
- sustained = {SUSTAINED} viewpoints

The key comparison is:
1. viewpoint-only baseline
2. A-only
3. A + viewpoint
4. A + C + viewpoint
5. trajectory model

Cross-validation is grouped by expression folder to reduce identity/expression
leakage between train and test.

Permutation testing evaluates whether A-before-B ordering is unusually high
under the tested null.

Do not interpret significance as proof of causality.
"""

    (OUT/"README.md").write_text(readme, encoding="utf-8")

    print("\n" + "#"*70)
    print("DONE")
    print("#"*70)
    print("Complete sequences:", len(events))
    for _,r in bs.iterrows():
        print(
            f"{r.side.upper():5s} "
            f"A before B={100*r.A_before_B:.3f}% "
            f"median A={r.median_A_distance:.1f}° "
            f"median B={r.median_B_distance:.1f}°"
        )
    for _,r in perm.iterrows():
        print(
            f"{r.side.upper():5s} permutation p="
            f"{r.p_permutation:.6g}"
        )
    if len(base) and len(primary):
        print(f"Viewpoint-only AUC : {base.iloc[0]:.4f}")
        print(f"A+C+viewpoint AUC : {primary.iloc[0]:.4f}")
        print(f"AUC gain           : {gain:.4f}")
    print("\nOutputs:", OUT)


if __name__ == "__main__":
    main()
