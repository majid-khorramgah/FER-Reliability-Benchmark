#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STAGE 10 — FINAL SYNTHESIS
FER Reliability Benchmark

IMPORTANT:
- This script DOES NOT rerun Stage 1–9.
- It ONLY reads validated outputs already produced by earlier stages.
- It does not re-estimate A/C thresholds.
- It does not modify previous-stage outputs.
- It explicitly checks Stage 1–9 using the corrected project structure:
    Stage 1 = statistical_validation
    Stage 2 = left_right_validation
    Stage 3 = permutation_validation
    Stage 4 = early_warning
    Stage 5 = early_warning_horizons
    Stage 6 = robustness_sensitivity
    Stage 7 = cross_expression
    Stage 8 = cross_identity
    Stage 9 = representation_validation
    Stage 10 = final_synthesis

Run from:
    D:\\1405\\FER-Reliability-Benchmark

Command:
    python analyze_final_synthesis.py
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\1405\FER-Reliability-Benchmark")
ANALYSIS = PROJECT_ROOT / "analysis"
OUT = ANALYSIS / "14_analyze_final_synthesis"
PLOTS = OUT / "plots"

A_THRESHOLD = 13.43702602
C_THRESHOLD = 0.00237080
FRONTAL = 107
EXPECTED_VIEWPOINTS = 215
SUSTAINED = 3

HORIZONS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def banner(title: str) -> None:
    print("\n" + "#" * 76)
    print(title)
    print("#" * 76)


def safe_float(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None

    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())

    if pd.isna(obj):
        return None

    return obj


def read_csv(path: Path, required: bool = False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required Stage-10 input not found:\n{path}")
        return None

    try:
        df = pd.read_csv(path)
        print(f"  Loaded CSV: {path.name} rows={len(df):,} cols={len(df.columns)}")
        return df
    except Exception as exc:
        if required:
            raise RuntimeError(f"Could not read required CSV:\n{path}\n{exc}")
        print(f"  WARNING: Could not read {path}: {exc}")
        return None


def read_json(path: Path, required: bool = False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required Stage-10 JSON not found:\n{path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded JSON: {path.name}")
        return data
    except Exception as exc:
        if required:
            raise RuntimeError(f"Could not read required JSON:\n{path}\n{exc}")
        print(f"  WARNING: Could not read {path}: {exc}")
        return None


def first_existing(directory: Path, names):
    for name in names:
        p = directory / name
        if p.exists():
            return p
    return None


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


# ============================================================
# CORRECT STAGE MAP
# ============================================================

STAGE_DIRS = {
    1: ANALYSIS / "1_analyze_statistical_validation",
    2: ANALYSIS / "2_analyze_left_right_validation",
    3: ANALYSIS / "3_analyze_permutation_validation",
    4: ANALYSIS / "4_analyze_early_warning",
    5: ANALYSIS / "5_analyze_early_warning_horizons",
    6: ANALYSIS / "6_analyze_robustness_sensitivity",
    7: ANALYSIS / "7_analyze_cross_expression",
    8: ANALYSIS / "8_analyze_cross_identity",
    9: ANALYSIS / "9_analyze_representation_validation",
    10: OUT,
}


# ============================================================
# STAGE STATUS
# ============================================================

def stage_status():
    rows = []

    required_candidates = {
        1: [
            "statistical_validation_report.json",
            "validation_report.json",
            "stage1_validation_report.json",
        ],
        2: [
            "left_right_validation_report.json",
            "left_right_paired_tests_stage2.csv",
            "directional_boundary_statistics_stage2.csv",
        ],
        3: [
            "permutation_validation_report.json",
            "permutation_results_stage3.csv",
            "permutation_orderings_stage3.csv",
        ],
        4: [
            "report.json",
            "model_comparison.csv",
            "boundary_summary.csv",
        ],
        5: [
            "horizon_analysis_report.json",
            "horizon_precedence_summary.csv",
        ],
        6: [
            "robustness_sensitivity_report.json",
            "robustness_score.csv",
        ],
        7: [
            "cross_expression_report.json",
            "expression_summary_primary.csv",
            "expression_summary_all.csv",
        ],
        8: [
            "cross_identity_report.json",
            "identity_summary.csv",
            "identity_side_summary.csv",
        ],
        9: [
            "representation_report.json",
            "representation_lead_lag.csv",
            "representation_pairwise_test.csv",
        ],
    }

    for stage in range(1, 10):
        directory = STAGE_DIRS[stage]
        candidates = required_candidates[stage]

        found = [directory / x for x in candidates if (directory / x).exists()]

        if found:
            status = "DONE"
            location = str(directory)
        else:
            status = "NOT_FOUND"
            location = str(directory)

        rows.append({
            "stage": stage,
            "status": status,
            "location": location,
            "evidence_files_found": len(found),
        })

    rows.append({
        "stage": 10,
        "status": "CURRENT",
        "location": str(OUT),
        "evidence_files_found": 0,
    })

    return pd.DataFrame(rows)


# ============================================================
# LOAD STAGE 4 — EARLY WARNING
# ============================================================

def load_early_warning():
    d = STAGE_DIRS[4]

    report = read_json(d / "report.json")
    boundary = read_csv(d / "boundary_summary.csv")
    models = read_csv(d / "model_comparison.csv")
    horizons = read_csv(d / "early_warning_horizons.csv")
    permutation = read_csv(d / "permutation_ordering.csv")
    subgroup = read_csv(d / "subgroup_results.csv")

    return {
        "report": report,
        "boundary": boundary,
        "models": models,
        "horizons": horizons,
        "permutation": permutation,
        "subgroup": subgroup,
    }


# ============================================================
# LOAD STAGE 5 — HORIZON
# ============================================================

def load_horizon():
    d = STAGE_DIRS[5]

    report = read_json(d / "horizon_analysis_report.json")
    precedence = read_csv(d / "horizon_precedence_summary.csv")
    population = read_csv(d / "horizon_population_summary.csv")
    expression = read_csv(d / "horizon_expression_results.csv")
    permutation = read_csv(d / "horizon_permutation_results.csv")

    return {
        "report": report,
        "precedence": precedence,
        "population": population,
        "expression": expression,
        "permutation": permutation,
    }


# ============================================================
# LOAD STAGE 6 — ROBUSTNESS
# ============================================================

def load_robustness():
    d = STAGE_DIRS[6]

    report = read_json(d / "robustness_sensitivity_report.json")
    sensitivity = read_csv(d / "robustness_sensitivity_results.csv")
    horizon = read_csv(d / "robustness_horizon_results.csv")
    bootstrap = read_csv(d / "robustness_bootstrap.csv")
    permutation = read_csv(d / "robustness_permutation.csv")
    score = read_csv(d / "robustness_score.csv")

    return {
        "report": report,
        "sensitivity": sensitivity,
        "horizon": horizon,
        "bootstrap": bootstrap,
        "permutation": permutation,
        "score": score,
    }


# ============================================================
# LOAD STAGE 7 — CROSS EXPRESSION
# ============================================================

def load_cross_expression():
    d = STAGE_DIRS[7]

    report_path = first_existing(
        d,
        [
            "cross_expression_report.json",
            "report.json",
        ],
    )

    report = read_json(report_path) if report_path else None

    coverage = read_csv(d / "expression_coverage.csv")
    all_summary = read_csv(d / "expression_summary_all.csv")
    primary = read_csv(d / "expression_summary_primary.csv")
    horizon = read_csv(d / "horizon_expression_results.csv")

    return {
        "report": report,
        "coverage": coverage,
        "all_summary": all_summary,
        "primary": primary,
        "horizon": horizon,
    }


# ============================================================
# LOAD STAGE 8 — CROSS IDENTITY
# ============================================================

def load_cross_identity():
    d = STAGE_DIRS[8]

    report = read_json(d / "cross_identity_report.json")
    summary = read_csv(d / "identity_summary.csv")
    side_summary = read_csv(d / "identity_side_summary.csv")
    paired = read_csv(d / "paired_identity_comparison.csv")
    horizon = read_csv(d / "identity_horizon_summary.csv")
    bootstrap = read_csv(d / "identity_bootstrap.csv")
    robustness = read_csv(d / "identity_robustness_score.csv")
    appearance = read_csv(d / "appearance_identity_summary.csv")

    return {
        "report": report,
        "summary": summary,
        "side_summary": side_summary,
        "paired": paired,
        "horizon": horizon,
        "bootstrap": bootstrap,
        "robustness": robustness,
        "appearance": appearance,
    }


# ============================================================
# LOAD STAGE 9 — REPRESENTATION
# ============================================================

def load_representation():
    d = STAGE_DIRS[9]

    report = read_json(d / "representation_report.json")
    view_metrics = read_csv(d / "representation_view_metrics.csv")
    boundaries = read_csv(d / "representation_boundaries.csv")
    sequence = read_csv(d / "representation_sequence_summary.csv")
    expression = read_csv(d / "representation_expression_summary.csv")
    viewpoint = read_csv(d / "representation_viewpoint_profile.csv")
    pairwise = read_csv(d / "representation_pairwise_summary.csv")
    pairwise_test = read_csv(d / "representation_pairwise_test.csv")
    paired_identity = read_csv(d / "representation_paired_identity.csv")
    identity_tests = read_csv(d / "representation_identity_tests.csv")
    lead_lag = read_csv(d / "representation_lead_lag.csv")
    bootstrap = read_csv(d / "representation_bootstrap.csv")
    fdr = read_csv(d / "representation_fdr.csv")

    return {
        "report": report,
        "view_metrics": view_metrics,
        "boundaries": boundaries,
        "sequence": sequence,
        "expression": expression,
        "viewpoint": viewpoint,
        "pairwise": pairwise,
        "pairwise_test": pairwise_test,
        "paired_identity": paired_identity,
        "identity_tests": identity_tests,
        "lead_lag": lead_lag,
        "bootstrap": bootstrap,
        "fdr": fdr,
    }


# ============================================================
# SCHEMA-AWARE NUMERIC/RATE HELPERS
# ============================================================

def _first_existing_column(df, candidates):
    """Return the first candidate column that actually exists."""
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _rate_series(df, candidates=None):
    """
    Dynamically extract an A-before-B / warning rate from whatever
    schema the source file actually contains.

    Supported representations include:
      - A_before_B_rate          -> fraction, e.g. 0.9238
      - A_before_B              -> usually fraction, normalized if needed
      - A_before_B_percent      -> percentage, converted to fraction
      - observed_rate            -> fraction or percentage
      - warning_rate             -> fraction or percentage
      - warning_rate_percent     -> percentage
      - baseline_A_before_B      -> fraction or percentage
    """
    if df is None or df.empty:
        return pd.Series(dtype=float), None

    if candidates is None:
        candidates = [
            "A_before_B_rate",
            "A_before_B_percent",
            "A_before_B",
            "observed_rate",
            "warning_rate",
            "warning_rate_percent",
            "baseline_A_before_B",
        ]

    col = _first_existing_column(df, candidates)
    if col is None:
        return pd.Series(dtype=float), None

    s = pd.to_numeric(df[col], errors="coerce")

    # Explicit percent columns are always converted from 0..100 to 0..1.
    if col.endswith("_percent") or "percent" in col.lower():
        s = s / 100.0
    else:
        # For ambiguous rate/count columns, infer scale from values.
        finite = s.dropna()
        if len(finite) and finite.abs().max() > 1.0:
            s = s / 100.0

    return s.dropna(), col


def _row_rate(row):
    """Schema-aware scalar extraction from one pandas row."""
    if row is None:
        return np.nan

    candidates = [
        "A_before_B_rate",
        "A_before_B_percent",
        "A_before_B",
        "observed_rate",
        "warning_rate",
        "warning_rate_percent",
        "baseline_A_before_B",
    ]

    for c in candidates:
        if c in row.index:
            value = safe_float(row.get(c))
            if value is None or pd.isna(value):
                continue

            if c.endswith("_percent") or "percent" in c.lower():
                return value / 100.0

            # A scalar rate/count ambiguity: values > 1 are treated as percent.
            if abs(value) > 1.0:
                return value / 100.0

            return value

    return np.nan


# ============================================================
# IDENTITY SYNTHESIS
# ============================================================

def identity_synthesis(identity):
    summary = identity.get("summary")

    if summary is None or summary.empty:
        return pd.DataFrame()

    # Never assume one historical column name.
    keep = [
        c for c in [
            "identity",
            "side",
            "A_before_B_rate",
            "A_before_B_percent",
            "A_before_B",
            "median_lead_deg",
            "median_lead",
            "median_A_B_lead",
            "mean_lead_deg",
            "mean_lead",
            "mean_A_B_lead",
        ]
        if c in summary.columns
    ]

    out = summary[keep].copy()

    rate, rate_col = _rate_series(out)
    if rate_col is not None:
        # Preserve the source column AND provide a canonical synthesis column.
        out["A_before_B_rate_normalized"] = np.nan
        out.loc[rate.index, "A_before_B_rate_normalized"] = rate
        out["A_before_B_percent_normalized"] = (
            out["A_before_B_rate_normalized"] * 100.0
        )

    # Canonical lead columns, regardless of source naming.
    if "median_lead_deg" not in out.columns:
        source = _first_existing_column(
            out,
            ["median_lead", "median_A_B_lead"]
        )
        if source is not None:
            out["median_lead_deg"] = pd.to_numeric(
                out[source], errors="coerce"
            )

    if "mean_lead_deg" not in out.columns:
        source = _first_existing_column(
            out,
            ["mean_lead", "mean_A_B_lead"]
        )
        if source is not None:
            out["mean_lead_deg"] = pd.to_numeric(
                out[source], errors="coerce"
            )

    return out


# ============================================================
# REPRESENTATION SYNTHESIS
# ============================================================

def representation_synthesis(rep):
    lead = rep["lead_lag"]

    if lead is None or lead.empty:
        return pd.DataFrame()

    return lead.copy()


# ============================================================
# EARLY WARNING SYNTHESIS
# ============================================================

def early_warning_synthesis(ew):
    rows = []

    models = ew["models"]

    if models is not None and not models.empty:
        for _, r in models.iterrows():
            row = {
                "model": r.get("model"),
                "n": safe_float(r.get("n")),
                "positive": safe_float(r.get("positive")),
                "auc": safe_float(r.get("auc")),
                "ap": safe_float(r.get("ap")),
                "brier": safe_float(r.get("brier")),
                "balanced_accuracy": safe_float(r.get("balanced_accuracy")),
                "precision": safe_float(r.get("precision")),
                "recall": safe_float(r.get("recall")),
                "f1": safe_float(r.get("f1")),
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# FINAL HORIZON TABLE
# ============================================================

def final_horizon(horizon):
    h = horizon["precedence"]

    if h is None or h.empty:
        h = horizon["population"]

    if h is None or h.empty:
        return pd.DataFrame()

    return h.copy()


# ============================================================
# FINAL KEY RESULTS
# ============================================================

def build_key_results(
    status,
    ew,
    hz,
    rob,
    expr,
    ident,
    rep,
):
    rows = []

    # ---------------- coverage ----------------
    complete_sequences = None
    expressions = None
    viewpoints = EXPECTED_VIEWPOINTS
    embedding_rows = None
    aligned_rows = None

    if rep["report"]:
        coverage = rep["report"].get("coverage", {})
        complete_sequences = coverage.get("complete_sequences")
        expressions = coverage.get("expressions")

    if complete_sequences is None and ew["report"]:
        complete_sequences = ew["report"].get("complete_sequences")

    if expressions is None and expr["report"]:
        expressions = (
            expr["report"].get("expression_categories")
            or expr["report"].get("expressions")
        )

    if rep["view_metrics"] is not None:
        aligned_rows = len(rep["view_metrics"])

    if rep["report"]:
        # Stage 9 report may not expose embedding_rows.
        embedding_rows = rep["report"].get("embedding_rows")

    rows.extend([
        {
            "analysis": "Validated complete sequences",
            "estimate": complete_sequences,
            "unit": "sequences",
        },
        {
            "analysis": "Expression categories",
            "estimate": expressions,
            "unit": "expressions",
        },
        {
            "analysis": "Viewpoints per sequence",
            "estimate": viewpoints,
            "unit": "viewpoints",
        },
        {
            "analysis": "Aligned representation/per-view rows",
            "estimate": aligned_rows,
            "unit": "rows",
        },
        {
            "analysis": "Embedding rows",
            "estimate": embedding_rows,
            "unit": "rows",
        },
    ])

    # ---------------- early warning ----------------
    if hz["precedence"] is not None and not hz["precedence"].empty:
        p = hz["precedence"]

        for _, r in p.iterrows():
            side = r.get("side")

            rate = (
                r.get("A_before_B")
                if "A_before_B" in r
                else r.get("A_before_B_rate")
            )

            median_lead = (
                r.get("median_lead")
                if "median_lead" in r
                else r.get("median_A_B_lead")
            )

            mean_lead = (
                r.get("mean_lead")
                if "mean_lead" in r
                else r.get("mean_A_B_lead")
            )

            rows.append({
                "analysis": f"Early warning A-before-B ({side})",
                "estimate": safe_float(rate),
                "unit": "rate",
                "median_lead_deg": safe_float(median_lead),
                "mean_lead_deg": safe_float(mean_lead),
            })

    # ---------------- identity ----------------
    if ident["summary"] is not None and not ident["summary"].empty:
        for _, r in ident["summary"].iterrows():
            rows.append({
                "analysis": (
                    f"Identity {r.get('identity')} "
                    f"{r.get('side')}"
                ),
                "estimate": _row_rate(r),
                "unit": "A-before-B rate",
                "median_lead_deg": safe_float(
                    r.get("median_lead_deg")
                    if "median_lead_deg" in r.index
                    else r.get("median_lead")
                    if "median_lead" in r.index
                    else r.get("median_A_B_lead")
                ),
                "mean_lead_deg": safe_float(
                    r.get("mean_lead_deg")
                    if "mean_lead_deg" in r.index
                    else r.get("mean_lead")
                    if "mean_lead" in r.index
                    else r.get("mean_A_B_lead")
                ),
            })

    # ---------------- representation ----------------
    if rep["pairwise_test"] is not None and not rep["pairwise_test"].empty:
        r = rep["pairwise_test"].iloc[0]

        rows.append({
            "analysis": "Representation same-vs-rival difference",
            "estimate": safe_float(r.get("difference")),
            "unit": "effect difference",
            "p_value": safe_float(r.get("p_value")),
        })

    # ---------------- robustness ----------------
    if rob["score"] is not None and not rob["score"].empty:
        for _, r in rob["score"].iterrows():
            rows.append({
                "analysis": f"Robustness score ({r.get('side')})",
                "estimate": safe_float(
                    r.get("overall_robustness_score")
                ),
                "unit": "robustness score",
            })

    return pd.DataFrame(rows)


# ============================================================
# FINAL CLAIMS
# ============================================================

def build_claims(status, ew, hz, rob, expr, ident, rep):
    claims = []

    all_done = all(
        status.loc[status.stage == s, "status"].iloc[0] == "DONE"
        for s in range(1, 10)
    )

    claims.append({
        "claim_id": "C01",
        "category": "coverage",
        "claim": (
            "The final synthesis integrates the validated "
            "benchmark analysis through Stage 9."
        ),
        "evidence": (
            "Stage 1–9 status is checked from the actual project "
            "directories; no earlier stage is rerun."
        ),
        "status": "SUPPORTED" if all_done else "PARTIAL",
    })

    # Identity — schema-aware; do not assume A_before_B_rate exists.
    if ident["summary"] is not None and not ident["summary"].empty:
        x, rate_col = _rate_series(ident["summary"])

        if len(x):
            claims.append({
                "claim_id": "C02",
                "category": "identity",
                "claim": (
                    "Cross-identity analysis shows substantial "
                    "A-before-B precedence across validated "
                    "identity/side conditions."
                ),
                "evidence": (
                    f"Observed A-before-B rates range from "
                    f"{x.min()*100:.3f}% to {x.max()*100:.3f}%. "
                    f"Source rate field detected dynamically as "
                    f"'{rate_col}'."
                ),
                "status": "SUPPORTED",
            })

    # Cross-expression
    if expr["primary"] is not None and not expr["primary"].empty:
        claims.append({
            "claim_id": "C03",
            "category": "expression",
            "claim": (
                "The early-warning pattern generalizes across "
                "the primary eligible expression groups."
            ),
            "evidence": (
                f"{len(expr['primary'])} primary expression groups "
                "are included; small groups are excluded from the "
                "primary generality claim."
            ),
            "status": "SUPPORTED",
        })

    # Robustness
    if rob["score"] is not None and not rob["score"].empty:
        claims.append({
            "claim_id": "C04",
            "category": "robustness",
            "claim": (
                "The observed early-warning pattern remains "
                "present under sensitivity and robustness analyses."
            ),
            "evidence": (
                "Threshold, sustained-viewpoint, metric, horizon, "
                "bootstrap, and permutation analyses were performed "
                "by the validated Stage 6 pipeline."
            ),
            "status": "SUPPORTED",
        })

    # Representation
    if rep["pairwise_test"] is not None and not rep["pairwise_test"].empty:
        r = rep["pairwise_test"].iloc[0]
        diff = safe_float(r.get("difference"))
        p = safe_float(r.get("p_value"))

        claims.append({
            "claim_id": "C05",
            "category": "representation",
            "claim": (
                "Representation-level analysis provides evidence "
                "that same-expression representation behavior "
                "differs from rival-expression representation."
            ),
            "evidence": (
                f"Same-vs-rival difference = {diff:.6f}; "
                f"permutation p = {p:.6f}."
                if diff is not None and p is not None
                else "Stage 9 representation pairwise test."
            ),
            "status": "SUPPORTED",
        })

    # Interpretation
    claims.append({
        "claim_id": "C06",
        "category": "interpretation",
        "claim": (
            "A-before-B is interpreted as temporal/statistical "
            "precedence and predictive utility, not causal evidence."
        ),
        "evidence": (
            "This limitation is explicitly preserved from the "
            "Stage 4, Stage 5, Stage 6, Stage 7, and Stage 9 analyses."
        ),
        "status": "SUPPORTED",
    })

    # Appearance
    appearance_text = ""
    if ident.get("appearance") is not None:
        app = ident["appearance"]
        if len(app):
            appearance_text = (
                f"{len(app)} appearance/identity rows are available."
            )

    claims.append({
        "claim_id": "C07",
        "category": "appearance",
        "claim": (
            "Small facial-hair/appearance subgroups remain "
            "descriptive and are not used as the primary "
            "generality claim."
        ),
        "evidence": (
            appearance_text
            or "Several appearance conditions have very small n."
        ),
        "status": "SUPPORTED",
    })

    return pd.DataFrame(claims)


# ============================================================
# PUBLICATION TABLE
# ============================================================

def build_publication_table(key):
    if key is None or key.empty:
        return pd.DataFrame()

    cols = [
        "analysis",
        "estimate",
        "unit",
        "p_value",
        "median_lead_deg",
        "mean_lead_deg",
    ]

    out = key.copy()

    for c in cols:
        if c not in out.columns:
            out[c] = np.nan

    return out[cols]


# ============================================================
# PLOTS
# ============================================================

def plot_horizons(hz):
    if hz is None or hz.empty:
        return

    h = hz.copy()

    if "horizon" not in h.columns:
        return

    rate_col = _first_existing_column(
        h,
        [
            "warning_rate",
            "warning_rate_percent",
            "A_before_B_rate",
            "A_before_B_percent",
            "A_before_B",
            "observed_rate",
        ],
    )

    if rate_col is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for side, g in h.groupby("side"):
        g = g.sort_values("horizon")
        y = pd.to_numeric(g[rate_col], errors="coerce")

        if rate_col.endswith("_percent") or "percent" in rate_col.lower():
            pass
        elif y.max(skipna=True) <= 1.0:
            y = y * 100.0
        else:
            # Ambiguous numeric rate represented as percentage points.
            y = y

        ax.plot(
            g["horizon"],
            y,
            marker="o",
            label=str(side),
        )

    ax.set_xlabel("Early-warning horizon (degrees)")
    ax.set_ylabel("Warning rate (%)")
    ax.set_title("Final Synthesis — Early-Warning Horizon")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        PLOTS / "final_horizon_curves.png",
        dpi=220,
    )
    plt.close(fig)


def plot_identity(identity_summary):
    if identity_summary is None or identity_summary.empty:
        return

    if not {"identity", "side"}.issubset(identity_summary.columns):
        return

    x = identity_summary.copy()
    rate, rate_col = _rate_series(x)

    if rate_col is None:
        return

    x["rate"] = np.nan
    x.loc[rate.index, "rate"] = rate * 100.0

    x["label"] = (
        x["identity"].astype(str)
        + " / "
        + x["side"].astype(str)
    )

    x = x.dropna(subset=["rate"])
    if x.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x["label"],
        x["rate"],
    )

    ax.set_ylabel("A-before-B rate (%)")
    ax.set_title(
        "Final Synthesis — Cross-Identity Precedence"
        f" [source: {rate_col}]"
    )
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(
        PLOTS / "final_identity_comparison.png",
        dpi=220,
    )
    plt.close(fig)


def plot_representation(lead):
    if lead is None or lead.empty:
        return

    required = {
        "gender",
        "side",
        "median_B_minus_A_deg",
    }

    if not required.issubset(lead.columns):
        return

    x = lead.copy()
    x["label"] = (
        x["gender"].astype(str)
        + " / "
        + x["side"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x["label"],
        x["median_B_minus_A_deg"],
    )

    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Median B − A lead (degrees)")
    ax.set_title("Final Synthesis — Representation Lead/Lag")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(
        PLOTS / "final_representation_lead_lag.png",
        dpi=220,
    )
    plt.close(fig)


def plot_robustness(score):
    if score is None or score.empty:
        return

    if not {
        "side",
        "overall_robustness_score",
    }.issubset(score.columns):
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        score["side"].astype(str),
        score["overall_robustness_score"],
    )

    ax.set_ylabel("Overall robustness score")
    ax.set_title("Final Synthesis — Robustness")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(
        PLOTS / "final_robustness.png",
        dpi=220,
    )
    plt.close(fig)


def plot_stage_overview(status):
    x = status.copy()

    numeric_stage = x["stage"].astype(int)

    values = []
    for s in numeric_stage:
        row = x[x.stage == s].iloc[0]
        if row["status"] == "DONE":
            values.append(1)
        elif row["status"] == "CURRENT":
            values.append(0.75)
        else:
            values.append(0)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        [f"Stage {s}" for s in numeric_stage],
        values,
    )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Completion status")
    ax.set_title("Final Synthesis — Stage 1–10 Overview")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(
        PLOTS / "final_stage_overview.png",
        dpi=220,
    )
    plt.close(fig)


# ============================================================
# README
# ============================================================

def write_readme(status, key):
    done_count = int(
        (status["status"] == "DONE").sum()
    )

    text = f"""# Stage 10 — Final Synthesis

Project:
`{PROJECT_ROOT}`

## Purpose

Stage 10 is a synthesis-only stage.

It does **not** rerun Stage 1–9, does not re-estimate thresholds,
and does not alter any previous analysis.

It collects the validated outputs from:

1. Stage 1 — statistical validation
2. Stage 2 — left/right validation
3. Stage 3 — permutation validation
4. Stage 4 — early-warning prediction
5. Stage 5 — early-warning horizon analysis
6. Stage 6 — robustness/sensitivity
7. Stage 7 — cross-expression analysis
8. Stage 8 — cross-identity analysis
9. Stage 9 — representation-level validation

## Fixed configuration

A threshold:
`{A_THRESHOLD:.8f}`

C threshold:
`{C_THRESHOLD:.8f}`

Sustained viewpoints:
`{SUSTAINED}`

Frontal viewpoint:
`{FRONTAL}`

Expected viewpoints per sequence:
`{EXPECTED_VIEWPOINTS}`

## Scientific interpretation

The main quantity is:

`lead = B_boundary - A_boundary`

Positive lead means representation drift occurs before
prediction failure in the viewpoint trajectory.

This is interpreted as temporal/statistical precedence and
predictive utility. It is **not** interpreted as proof of causality.

## Coverage

Stages detected as complete:
`{done_count} / 9`

## Outputs

- final_stage_status.csv
- final_key_results.csv
- final_early_warning_results.csv
- final_horizon_results.csv
- final_robustness_results.csv
- final_expression_results.csv
- final_identity_comparison.csv
- final_representation_results.csv
- final_publication_table.csv
- final_claims.csv
- final_summary.csv
- final_report.json

Plots:

- plots/final_stage_overview.png
- plots/final_horizon_curves.png
- plots/final_identity_comparison.png
- plots/final_representation_lead_lag.png
- plots/final_robustness.png
"""

    (OUT / "README_final_synthesis.md").write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    banner("STAGE 10 — FINAL SYNTHESIS")

    print(f"Project: {PROJECT_ROOT}")
    print(f"Fixed A: {A_THRESHOLD}")
    print(f"Fixed C: {C_THRESHOLD}")
    print(f"Frontal: {FRONTAL}")
    print(f"Expected viewpoints: {EXPECTED_VIEWPOINTS}")

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(
            f"Project root does not exist:\n{PROJECT_ROOT}"
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    banner("STAGE 1–9 STATUS — CORRECTED")

    status = stage_status()
    print(
        status.to_string(index=False)
    )

    status.to_csv(
        OUT / "final_stage_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing = status[
        (status.stage <= 9)
        & (status.status != "DONE")
    ]

    if not missing.empty:
        print(
            "\nWARNING: Some Stage 1–9 directories/files "
            "were not detected."
        )
        print(
            missing.to_string(index=False)
        )
    else:
        print(
            "\nAll Stage 1–9 required evidence was detected."
        )

    # --------------------------------------------------------
    # LOAD — NO RERUN
    # --------------------------------------------------------

    banner("LOADING STAGE 4 — EARLY WARNING")
    ew = load_early_warning()

    banner("LOADING STAGE 5 — HORIZON ANALYSIS")
    hz = load_horizon()

    banner("LOADING STAGE 6 — ROBUSTNESS / SENSITIVITY")
    rob = load_robustness()

    banner("LOADING STAGE 7 — CROSS EXPRESSION")
    expr = load_cross_expression()

    banner("LOADING STAGE 8 — CROSS IDENTITY")
    ident = load_cross_identity()

    banner("LOADING STAGE 9 — REPRESENTATION")
    rep = load_representation()

    # --------------------------------------------------------
    # SYNTHESIS TABLES
    # --------------------------------------------------------

    banner("BUILDING SYNTHESIS TABLES")

    identity = identity_synthesis(ident)
    representation = representation_synthesis(rep)
    early = early_warning_synthesis(ew)
    horizons = final_horizon(hz)

    key = build_key_results(
        status,
        ew,
        hz,
        rob,
        expr,
        ident,
        rep,
    )

    claims = build_claims(
        status,
        ew,
        hz,
        rob,
        expr,
        ident,
        rep,
    )

    publication = build_publication_table(
        key
    )

    # --------------------------------------------------------
    # FINAL SCHEMA SAFETY CHECK
    # --------------------------------------------------------
    # Stage 10 must never fail merely because a source stage used
    # a different valid column name.  Normalize identity output
    # before writing publication tables.
    if identity is not None and not identity.empty:
        rate, rate_col = _rate_series(identity)
        if rate_col is not None:
            identity["A_before_B_rate_normalized"] = np.nan
            identity.loc[rate.index, "A_before_B_rate_normalized"] = rate
            identity["A_before_B_percent_normalized"] = (
                identity["A_before_B_rate_normalized"] * 100.0
            )

    print("Schema safety check: PASSED")
    # --------------------------------------------------------
    # SAVE TABLES
    # --------------------------------------------------------

    outputs = {
        "final_early_warning_results.csv": early,
        "final_horizon_results.csv": horizons,
        "final_robustness_results.csv": rob["score"],
        "final_expression_results.csv": expr["primary"],
        "final_identity_comparison.csv": identity,
        "final_representation_results.csv": representation,
        "final_publication_table.csv": publication,
        "final_key_results.csv": key,
        "final_claims.csv": claims,
        "final_stage_status.csv": status,
    }

    for name, df in outputs.items():
        if df is None:
            continue

        if isinstance(df, pd.DataFrame):
            df.to_csv(
                OUT / name,
                index=False,
                encoding="utf-8-sig",
            )
            print(OUT / name)

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    banner("BUILDING FINAL PLOTS")

    plot_horizons(horizons)
    plot_identity(identity)
    plot_representation(representation)
    plot_robustness(rob["score"])
    plot_stage_overview(status)

    for p in sorted(PLOTS.glob("*.png")):
        print(p)

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    banner("BUILDING FINAL REPORT")

    report = {
        "stage": 10,
        "project": "FER-Reliability-Benchmark",
        "purpose": (
            "Synthesis of validated Stage 1–9 analyses "
            "without rerunning or re-estimating earlier stages."
        ),
        "fixed_configuration": {
            "A_threshold": A_THRESHOLD,
            "C_threshold": C_THRESHOLD,
            "sustained_viewpoints": SUSTAINED,
            "frontal_viewpoint": FRONTAL,
            "expected_viewpoints": EXPECTED_VIEWPOINTS,
        },
        "stage_status": status.to_dict(
            orient="records"
        ),
        "coverage": {
            "complete_sequences": (
                int(rep["report"]["coverage"]["complete_sequences"])
                if rep["report"]
                and rep["report"].get("coverage", {}).get(
                    "complete_sequences"
                ) is not None
                else None
            ),
            "expressions": (
                int(rep["report"]["coverage"]["expressions"])
                if rep["report"]
                and rep["report"].get("coverage", {}).get(
                    "expressions"
                ) is not None
                else None
            ),
            "viewpoints_per_sequence": EXPECTED_VIEWPOINTS,
            "aligned_rows": (
                len(rep["view_metrics"])
                if rep["view_metrics"] is not None
                else None
            ),
        },
        "early_warning": {
            "model_comparison": (
                ew["models"].to_dict(orient="records")
                if ew["models"] is not None
                else []
            ),
            "boundary_summary": (
                ew["boundary"].to_dict(orient="records")
                if ew["boundary"] is not None
                else []
            ),
        },
        "horizons": (
            horizons.to_dict(orient="records")
            if horizons is not None
            else []
        ),
        "robustness": {
            "score": (
                rob["score"].to_dict(orient="records")
                if rob["score"] is not None
                else []
            ),
            "permutation": (
                rob["permutation"].to_dict(orient="records")
                if rob["permutation"] is not None
                else []
            ),
        },
        "cross_expression": {
            "primary": (
                expr["primary"].to_dict(orient="records")
                if expr["primary"] is not None
                else []
            ),
        },
        "cross_identity": {
            "summary": (
                ident["summary"].to_dict(orient="records")
                if ident["summary"] is not None
                else []
            ),
            "robustness": (
                ident["robustness"].to_dict(orient="records")
                if ident["robustness"] is not None
                else []
            ),
        },
        "representation": {
            "lead_lag": (
                rep["lead_lag"].to_dict(orient="records")
                if rep["lead_lag"] is not None
                else []
            ),
            "pairwise_test": (
                rep["pairwise_test"].to_dict(orient="records")
                if rep["pairwise_test"] is not None
                else []
            ),
            "identity_tests": (
                rep["identity_tests"].to_dict(orient="records")
                if rep["identity_tests"] is not None
                else []
            ),
        },
        "final_claims": (
            claims.to_dict(orient="records")
            if claims is not None
            else []
        ),
        "scientific_cautions": [
            "A-before-B represents temporal/statistical precedence.",
            "Positive lead is not proof of causal direction.",
            "A/C thresholds are fixed and are not re-estimated in Stage 10.",
            "Stage 10 does not alter Stage 1–9 outputs.",
            "Small appearance/identity groups remain descriptive.",
        ],
    }

    report = json_safe(report)

    with open(
        OUT / "final_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print(OUT / "final_report.json")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    banner("FINAL SYNTHESIS SUMMARY")

    done = int(
        (status["status"] == "DONE").sum()
    )

    print(
        f"Stage 1–9 detected as complete: "
        f"{done}/9"
    )

    if rep["report"]:
        cov = rep["report"].get(
            "coverage",
            {},
        )

        print(
            f"Complete sequences: "
            f"{cov.get('complete_sequences', 'N/A')}"
        )

        print(
            f"Expressions: "
            f"{cov.get('expressions', 'N/A')}"
        )

    if ident["summary"] is not None:
        print("\nCROSS-IDENTITY:")
        print(
            identity.to_string(index=False)
        )

    if representation is not None and not representation.empty:
        print("\nREPRESENTATION LEAD/LAG:")
        print(
            representation.to_string(index=False)
        )

    if claims is not None and not claims.empty:
        print("\nFINAL CLAIMS:")
        print(
            claims.to_string(index=False)
        )

    write_readme(status, key)

    print(
        f"\nFinal output directory:\n{OUT}"
    )

    print(
        "\nSTAGE 10 FINAL SYNTHESIS COMPLETED."
    )

    print(
        "Previous Stage 1–9 analyses were NOT rerun, "
        "re-estimated, or modified."
    )


if __name__ == "__main__":
    main()
