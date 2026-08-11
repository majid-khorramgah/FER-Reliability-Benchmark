from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# CROSS-EXPRESSION ANALYSIS
# VALIDATED / CONSISTENT WITH EARLY-WARNING PIPELINE
# ======================================================================
#
# Scientific question:
#
#   Does the early-warning phenomenon generalize across expressions,
#   or is it concentrated in a small subset of expressions?
#
# IMPORTANT:
#
# This script MUST use the same event definitions as:
#
#   analyze_early_warning.py
#   analyze_early_warning_horizons.py
#
# Therefore:
#
#   A threshold = 13.43702602
#   C threshold = 0.00237080
#   sustained viewpoints = 3
#
# These thresholds are NOT re-estimated per expression.
#
# A:
#   A_angular_distance_deg >= A_THRESHOLD
#   for 3 consecutive viewpoints
#
# C:
#   C_margin <= C_THRESHOLD
#   for 3 consecutive viewpoints
#
# B:
#   predicted_folder != true folder
#   for 3 consecutive viewpoints
#
# A positive A-before-B lead means:
#
#   representation drift boundary occurs farther BEFORE
#   prediction failure boundary.
#
# This is temporal/statistical precedence, NOT causality.
# ======================================================================


# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[0]
ANALYSIS = ROOT / "analysis"

INPUT = ANALYSIS / "4_analyze_embeddings_trajectory" / "per_view_metrics_multimetric.csv"

OUT = ANALYSIS / "11_analyze_cross_expression"
PLOTS = OUT / "plots"


# ----------------------------------------------------------------------
# VALIDATED CONFIGURATION
# ----------------------------------------------------------------------

FRONTAL = 107
EXPECTED_VIEWS = 215

A_THRESHOLD = 13.43702602
C_THRESHOLD = 0.00237080

SUSTAINED = 3

# Primary analysis requires enough independent sequences
# in an expression category.
MIN_PRIMARY_N = 8

HORIZONS = (
    1,
    2,
    3,
    5,
    7,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
)

N_PERM = 2000
N_BOOT = 2000

SEED = 1405

rng = np.random.default_rng(SEED)


# ======================================================================
# COLUMN DETECTION
# ======================================================================

def find_col(df, names, required=True):

    exact = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in names:
        if name.lower() in exact:
            return exact[name.lower()]

    normalized = {
        "".join(
            ch for ch in str(c).lower()
            if ch.isalnum()
        ): c
        for c in df.columns
    }

    for name in names:

        key = "".join(
            ch for ch in name.lower()
            if ch.isalnum()
        )

        if key in normalized:
            return normalized[key]

    if required:
        raise RuntimeError(
            f"Missing required column.\n"
            f"Tried: {names}\n"
            f"Available: {list(df.columns)}"
        )

    return None


def get_columns(df):

    return {
        "expression": find_col(
            df,
            ["expression"],
        ),

        "folder": find_col(
            df,
            ["folder"],
        ),

        "viewpoint": find_col(
            df,
            ["viewpoint", "angle"],
        ),

        "A": find_col(
            df,
            [
                "A_angular_distance_deg",
                "A_angular_distance",
                "A_angular",
            ],
        ),

        "C": find_col(
            df,
            ["C_margin"],
        ),

        "B": find_col(
            df,
            [
                "B_predicted_folder",
                "B_predicted",
                "B",
            ],
        ),
    }


def numeric(x):

    return pd.to_numeric(
        x,
        errors="coerce",
    )


# ======================================================================
# FIRST SUSTAINED CROSSING
# ======================================================================

def first_sustained(viewpoints, condition, side):

    v = np.asarray(
        viewpoints,
        dtype=float,
    )

    condition = np.asarray(
        condition,
        dtype=bool,
    )

    valid = np.isfinite(v)

    v = v[valid]
    condition = condition[valid]

    if side == "left":

        keep = v <= FRONTAL

        v = v[keep]
        condition = condition[keep]

        # 107 -> 106 -> ... -> 0
        order = np.argsort(v)[::-1]

    elif side == "right":

        keep = v >= FRONTAL

        v = v[keep]
        condition = condition[keep]

        # 107 -> 108 -> ... -> 214
        order = np.argsort(v)

    else:
        raise ValueError(
            f"Unknown side: {side}"
        )

    v = v[order]
    condition = condition[order]

    run = 0

    for i, ok in enumerate(condition):

        if ok:
            run += 1
        else:
            run = 0

        if run >= SUSTAINED:

            return float(
                v[i - SUSTAINED + 1]
            )

    return np.nan


# ======================================================================
# BUILD ONE EVENT ROW PER COMPLETE FOLDER
# ======================================================================

def build_events(df, c):

    rows = []

    groups = list(
        df.groupby(
            c["folder"],
            sort=False,
        )
    )

    print(
        f"Total folders: {len(groups)}"
    )

    for i, (folder, g) in enumerate(
        groups,
        start=1,
    ):

        work = g.copy()

        work["_vp"] = numeric(
            work[c["viewpoint"]]
        )

        vp = (
            work["_vp"]
            .dropna()
            .astype(int)
        )

        # --------------------------------------------------------------
        # COMPLETE SEQUENCE REQUIREMENT
        # --------------------------------------------------------------

        if (
            len(vp) != EXPECTED_VIEWS
            or set(vp) != set(
                range(EXPECTED_VIEWS)
            )
        ):
            continue

        # Always sort by viewpoint.
        work = work.sort_values(
            "_vp"
        )

        v = work["_vp"].to_numpy(
            dtype=float
        )

        A = numeric(
            work[c["A"]]
        ).to_numpy(
            dtype=float
        )

        C = numeric(
            work[c["C"]]
        ).to_numpy(
            dtype=float
        )

        # --------------------------------------------------------------
        # B FAILURE
        # --------------------------------------------------------------

        predicted = (
            work[c["B"]]
            .astype(str)
            .str.strip()
        )

        truth = str(folder)

        invalid_prediction = (
            predicted.str.lower().isin(
                {
                    "",
                    "nan",
                    "none",
                    "null",
                    "na",
                    "n/a",
                }
            )
        )

        B_failure = (
            (~invalid_prediction)
            & (predicted != truth)
        ).to_numpy(
            dtype=bool
        )

        expression = str(
            work[c["expression"]].iloc[0]
        )

        row = {
            "expression": expression,
            "folder": truth,
        }

        # --------------------------------------------------------------
        # LEFT / RIGHT
        # --------------------------------------------------------------

        for side in (
            "left",
            "right",
        ):

            row[f"A_{side}"] = (
                first_sustained(
                    v,
                    (
                        np.isfinite(A)
                        & (
                            A >= A_THRESHOLD
                        )
                    ),
                    side,
                )
            )

            row[f"C_{side}"] = (
                first_sustained(
                    v,
                    (
                        np.isfinite(C)
                        & (
                            C <= C_THRESHOLD
                        )
                    ),
                    side,
                )
            )

            row[f"B_{side}"] = (
                first_sustained(
                    v,
                    B_failure,
                    side,
                )
            )

        rows.append(row)

        if (
            i <= 5
            or i % 50 == 0
            or i == len(groups)
        ):
            print(
                f"Processed {i}/{len(groups)}"
            )

    return pd.DataFrame(rows)


# ======================================================================
# CONVERT BOUNDARIES TO DISTANCE FROM FRONTAL
# ======================================================================

def add_distances(events):

    x = events.copy()

    for boundary in (
        "A",
        "C",
        "B",
    ):

        for side in (
            "left",
            "right",
        ):

            source = (
                f"{boundary}_{side}"
            )

            target = (
                f"{boundary}_distance_{side}"
            )

            x[target] = (
                FRONTAL
                - x[source].astype(float)
            ).abs()

    for side in (
        "left",
        "right",
    ):

        A = x[
            f"A_distance_{side}"
        ]

        B = x[
            f"B_distance_{side}"
        ]

        C = x[
            f"C_distance_{side}"
        ]

        # Positive = A happens earlier.
        x[
            f"AB_lead_{side}"
        ] = B - A

        x[
            f"AC_lead_{side}"
        ] = C - A

        x[
            f"CB_lead_{side}"
        ] = B - C

    return x


# ======================================================================
# EXPRESSION SUMMARY
# ======================================================================

def summarize_expression(
    events,
    expression,
    side,
):

    g = events[
        events.expression == expression
    ]

    A = g[
        f"A_distance_{side}"
    ].to_numpy(float)

    B = g[
        f"B_distance_{side}"
    ].to_numpy(float)

    C = g[
        f"C_distance_{side}"
    ].to_numpy(float)

    ok_ab = (
        np.isfinite(A)
        & np.isfinite(B)
    )

    ok_ac = (
        np.isfinite(A)
        & np.isfinite(C)
    )

    ok_cb = (
        np.isfinite(C)
        & np.isfinite(B)
    )

    A_ab = A[ok_ab]
    B_ab = B[ok_ab]

    A_ac = A[ok_ac]
    C_ac = C[ok_ac]

    C_cb = C[ok_cb]
    B_cb = B[ok_cb]

    lead = (
        B_ab - A_ab
    )

    n = len(g)

    return {
        "expression": expression,
        "side": side,

        "n_sequences": n,

        "AB_n": len(A_ab),
        "AC_n": len(A_ac),
        "CB_n": len(C_cb),

        "primary_eligible": (
            n >= MIN_PRIMARY_N
        ),

        "A_before_B_percent": (
            100
            * np.mean(
                A_ab < B_ab
            )
            if len(A_ab)
            else np.nan
        ),

        "A_before_C_percent": (
            100
            * np.mean(
                A_ac < C_ac
            )
            if len(A_ac)
            else np.nan
        ),

        "C_before_B_percent": (
            100
            * np.mean(
                C_cb < B_cb
            )
            if len(C_cb)
            else np.nan
        ),

        "median_A_distance": (
            np.median(A_ab)
            if len(A_ab)
            else np.nan
        ),

        "median_C_distance": (
            np.median(C_ac)
            if len(C_ac)
            else np.nan
        ),

        "median_B_distance": (
            np.median(B_ab)
            if len(B_ab)
            else np.nan
        ),

        "median_A_B_lead": (
            np.median(lead)
            if len(lead)
            else np.nan
        ),

        "mean_A_B_lead": (
            np.mean(lead)
            if len(lead)
            else np.nan
        ),

        "positive_lead_percent": (
            100
            * np.mean(lead > 0)
            if len(lead)
            else np.nan
        ),
    }


def build_expression_summary(events):

    rows = []

    expressions = sorted(
        events.expression.unique()
    )

    for expression in expressions:

        for side in (
            "left",
            "right",
        ):

            rows.append(
                summarize_expression(
                    events,
                    expression,
                    side,
                )
            )

    return pd.DataFrame(rows)


# ======================================================================
# CROSS-EXPRESSION HORIZONS
# ======================================================================

def build_horizons(events):

    rows = []

    for expression in sorted(
        events.expression.unique()
    ):

        g = events[
            events.expression == expression
        ]

        for side in (
            "left",
            "right",
        ):

            A = g[
                f"A_distance_{side}"
            ].to_numpy(float)

            B = g[
                f"B_distance_{side}"
            ].to_numpy(float)

            ok = (
                np.isfinite(A)
                & np.isfinite(B)
            )

            A = A[ok]
            B = B[ok]

            if len(A) == 0:
                continue

            lead = B - A

            for h in HORIZONS:

                warning = (
                    (lead > 0)
                    & (lead <= h)
                )

                rows.append({

                    "expression": expression,

                    "side": side,

                    "n": len(A),

                    "horizon_degrees": h,

                    "warning_count": int(
                        warning.sum()
                    ),

                    "warning_rate": float(
                        warning.mean()
                    ),

                    "A_before_B_count": int(
                        (lead > 0).sum()
                    ),

                    "A_before_B_rate": float(
                        (lead > 0).mean()
                    ),

                    "median_lead": float(
                        np.median(lead)
                    ),

                    "primary_eligible": (
                        len(A)
                        >= MIN_PRIMARY_N
                    ),
                })

    return pd.DataFrame(rows)


# ======================================================================
# POOLED PRIMARY RESULT
# ======================================================================

def pooled_primary(events):

    rows = []

    expression_counts = (
        events.groupby(
            "expression"
        )
        .size()
    )

    eligible = set(
        expression_counts[
            expression_counts
            >= MIN_PRIMARY_N
        ].index
    )

    d = events[
        events.expression.isin(
            eligible
        )
    ]

    for side in (
        "left",
        "right",
    ):

        A = d[
            f"A_distance_{side}"
        ].to_numpy(float)

        B = d[
            f"B_distance_{side}"
        ].to_numpy(float)

        ok = (
            np.isfinite(A)
            & np.isfinite(B)
        )

        A = A[ok]
        B = B[ok]

        lead = B - A

        if len(lead) == 0:
            continue

        rows.append({

            "side": side,

            "primary_expression_groups": (
                len(eligible)
            ),

            "primary_sequences": len(
                lead
            ),

            "A_before_B_percent": (
                100
                * np.mean(
                    lead > 0
                )
            ),

            "median_lead": (
                np.median(lead)
            ),

            "mean_lead": (
                np.mean(lead)
            ),

            "positive_lead_percent": (
                100
                * np.mean(
                    lead > 0
                )
            ),
        })

    return pd.DataFrame(rows)


# ======================================================================
# BOOTSTRAP ACROSS EXPRESSION CATEGORIES
# ======================================================================

def bootstrap_expression_rates(
    summary,
    side,
):

    x = summary[
        (summary.side == side)
        & summary.primary_eligible
        & np.isfinite(
            summary.A_before_B_percent
        )
    ]

    if len(x) == 0:

        return {
            "side": side,
            "n_expression_groups": 0,
            "mean_rate": np.nan,
            "median_rate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    rates = (
        x.A_before_B_percent
        .to_numpy(float)
        / 100.0
    )

    boot = []

    for _ in range(N_BOOT):

        sample = rng.choice(
            rates,
            size=len(rates),
            replace=True,
        )

        boot.append(
            np.mean(sample)
        )

    return {

        "side": side,

        "n_expression_groups": len(
            rates
        ),

        "mean_rate": float(
            np.mean(rates)
        ),

        "median_rate": float(
            np.median(rates)
        ),

        "ci_low": float(
            np.percentile(
                boot,
                2.5,
            )
        ),

        "ci_high": float(
            np.percentile(
                boot,
                97.5,
            )
        ),
    }


# ======================================================================
# EXPRESSION HETEROGENEITY
# ======================================================================

def heterogeneity_test(
    events,
    side,
):

    expression_values = []

    for expression, g in (
        events.groupby("expression")
    ):

        if len(g) < MIN_PRIMARY_N:
            continue

        A = g[
            f"A_distance_{side}"
        ].to_numpy(float)

        B = g[
            f"B_distance_{side}"
        ].to_numpy(float)

        ok = (
            np.isfinite(A)
            & np.isfinite(B)
        )

        A = A[ok]
        B = B[ok]

        if len(A) < MIN_PRIMARY_N:
            continue

        rate = np.mean(
            B > A
        )

        expression_values.append(
            rate
        )

    if len(expression_values) < 2:

        return {
            "side": side,
            "n_expression_groups": len(
                expression_values
            ),
            "observed_range": np.nan,
            "null_mean_range": np.nan,
            "p_value": np.nan,
        }

    observed = (
        np.max(expression_values)
        - np.min(expression_values)
    )

    # ------------------------------------------------------------------
    # Correct null:
    #
    # Randomly redistribute the SEQUENCE-LEVEL binary A-before-B
    # outcomes among expression groups while preserving group sizes.
    # ------------------------------------------------------------------

    sequence_rates = []

    group_sizes = []

    for expression, g in (
        events.groupby("expression")
    ):

        if len(g) < MIN_PRIMARY_N:
            continue

        A = g[
            f"A_distance_{side}"
        ].to_numpy(float)

        B = g[
            f"B_distance_{side}"
        ].to_numpy(float)

        ok = (
            np.isfinite(A)
            & np.isfinite(B)
        )

        A = A[ok]
        B = B[ok]

        if len(A) < MIN_PRIMARY_N:
            continue

        sequence_rates.extend(
            (B > A).astype(int)
        )

        group_sizes.append(
            len(A)
        )

    sequence_rates = np.asarray(
        sequence_rates,
        dtype=int,
    )

    null_ranges = []

    for _ in range(N_PERM):

        shuffled = rng.permutation(
            sequence_rates
        )

        start = 0
        rates = []

        for size in group_sizes:

            block = shuffled[
                start:start + size
            ]

            rates.append(
                np.mean(block)
            )

            start += size

        null_ranges.append(
            np.max(rates)
            - np.min(rates)
        )

    null_ranges = np.asarray(
        null_ranges
    )

    p = (
        1
        + np.sum(
            null_ranges >= observed
        )
    ) / (
        N_PERM + 1
    )

    return {

        "side": side,

        "n_expression_groups": len(
            group_sizes
        ),

        "observed_range": float(
            observed
        ),

        "null_mean_range": float(
            np.mean(null_ranges)
        ),

        "p_value": float(p),

        "permutation_repetitions": (
            N_PERM
        ),
    }


# ======================================================================
# KRUSKAL-WALLIS
# ======================================================================

def run_kruskal(events, side):

    try:
        from scipy.stats import kruskal
    except ImportError:

        return {
            "side": side,
            "statistic": np.nan,
            "p_value": np.nan,
            "groups": 0,
            "note": (
                "scipy not installed"
            ),
        }

    groups = []

    for expression, g in (
        events.groupby("expression")
    ):

        if len(g) < MIN_PRIMARY_N:
            continue

        A = g[
            f"A_distance_{side}"
        ].to_numpy(float)

        B = g[
            f"B_distance_{side}"
        ].to_numpy(float)

        ok = (
            np.isfinite(A)
            & np.isfinite(B)
        )

        lead = (
            B[ok] - A[ok]
        )

        if len(lead) >= MIN_PRIMARY_N:
            groups.append(lead)

    if len(groups) < 2:

        return {
            "side": side,
            "statistic": np.nan,
            "p_value": np.nan,
            "groups": len(groups),
        }

    statistic, p = kruskal(
        *groups
    )

    return {

        "side": side,

        "statistic": float(
            statistic
        ),

        "p_value": float(p),

        "groups": len(groups),
    }


# ======================================================================
# LEAVE-ONE-EXPRESSION-OUT
# ======================================================================

def leave_one_out(events):

    rows = []

    for excluded in sorted(
        events.expression.unique()
    ):

        remaining = events[
            events.expression
            != excluded
        ]

        row = {
            "excluded_expression":
                excluded
        }

        for side in (
            "left",
            "right",
        ):

            A = remaining[
                f"A_distance_{side}"
            ].to_numpy(float)

            B = remaining[
                f"B_distance_{side}"
            ].to_numpy(float)

            ok = (
                np.isfinite(A)
                & np.isfinite(B)
            )

            A = A[ok]
            B = B[ok]

            if len(A) == 0:

                row[
                    f"A_before_B_{side}"
                ] = np.nan

                row[
                    f"median_lead_{side}"
                ] = np.nan

                continue

            lead = B - A

            row[
                f"A_before_B_{side}"
            ] = float(
                np.mean(
                    lead > 0
                )
            )

            row[
                f"median_lead_{side}"
            ] = float(
                np.median(lead)
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
# PLOTS
# ======================================================================

def plot_warning_rates(
    summary,
    side,
    output,
):

    x = summary[
        (summary.side == side)
        & summary.primary_eligible
    ].copy()

    x = x[
        np.isfinite(
            x.A_before_B_percent
        )
    ]

    if x.empty:
        return

    x = x.sort_values(
        "A_before_B_percent",
        ascending=False,
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        np.arange(len(x)),
        x.A_before_B_percent,
    )

    plt.axhline(
        50,
        linestyle="--",
        linewidth=1,
        label="50%",
    )

    plt.axhline(
        80,
        linestyle=":",
        linewidth=1,
        label="80%",
    )

    plt.xticks(
        np.arange(len(x)),
        x.expression,
        rotation=75,
        ha="right",
        fontsize=7,
    )

    plt.ylabel(
        "A-before-B (%)"
    )

    plt.xlabel(
        "Expression"
    )

    plt.title(
        f"Cross-expression A-before-B — {side}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=220,
    )

    plt.close()


def plot_leads(
    summary,
    side,
    output,
):

    x = summary[
        (summary.side == side)
        & summary.primary_eligible
        & np.isfinite(
            summary.median_A_B_lead
        )
    ].copy()

    if x.empty:
        return

    x = x.sort_values(
        "median_A_B_lead",
        ascending=False,
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        np.arange(len(x)),
        x.median_A_B_lead,
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xticks(
        np.arange(len(x)),
        x.expression,
        rotation=75,
        ha="right",
        fontsize=7,
    )

    plt.ylabel(
        "Median B distance − A distance (degrees)"
    )

    plt.xlabel(
        "Expression"
    )

    plt.title(
        f"Cross-expression early-warning lead — {side}"
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=220,
    )

    plt.close()


def plot_horizon_heatmap(
    horizon,
    output,
):

    x = horizon[
        horizon.primary_eligible
    ].copy()

    if x.empty:
        return

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 12),
    )

    for ax, side in zip(
        axes,
        ["left", "right"],
    ):

        d = x[
            x.side == side
        ]

        if d.empty:
            ax.axis("off")
            continue

        table = d.pivot_table(
            index="expression",
            columns="horizon_degrees",
            values="warning_rate",
            aggfunc="mean",
        )

        im = ax.imshow(
            table.to_numpy(),
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(
            np.arange(
                len(table.columns)
            )
        )

        ax.set_xticklabels(
            [
                f"{int(v)}°"
                for v in table.columns
            ]
        )

        ax.set_yticks(
            np.arange(
                len(table.index)
            )
        )

        ax.set_yticklabels(
            table.index,
            fontsize=7,
        )

        ax.set_xlabel(
            "Warning horizon"
        )

        ax.set_ylabel(
            "Expression"
        )

        ax.set_title(
            f"Cross-expression horizon — {side}"
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Warning rate",
        )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=220,
    )

    plt.close()


# ======================================================================
# REPORT
# ======================================================================

def write_report(
    df,
    events,
    summary,
    horizons,
    pooled,
    bootstrap,
    heterogeneity,
    kruskal,
    loo,
):

    report = {

        "project":
            "FER-Reliability-Benchmark",

        "analysis":
            "Cross-expression validated analysis",

        "scientific_question":
            (
                "Does the early-warning phenomenon "
                "generalize across expression categories?"
            ),

        "configuration": {

            "frontal":
                FRONTAL,

            "expected_views":
                EXPECTED_VIEWS,

            "A_threshold":
                A_THRESHOLD,

            "C_threshold":
                C_THRESHOLD,

            "sustained":
                SUSTAINED,

            "minimum_primary_expression_n":
                MIN_PRIMARY_N,

            "horizons":
                list(HORIZONS),

            "N_permutation":
                N_PERM,

            "N_bootstrap":
                N_BOOT,

            "seed":
                SEED,
        },

        "data": {

            "rows_loaded":
                int(len(df)),

            "complete_sequences":
                int(len(events)),

            "expression_categories":
                int(
                    events.expression.nunique()
                ),
        },

        "coverage":
            (
                events.groupby(
                    "expression"
                )
                .size()
                .rename("n_sequences")
                .reset_index()
                .to_dict(
                    orient="records"
                )
            ),

        "pooled":
            pooled.to_dict(
                orient="records"
            ),

        "bootstrap":
            bootstrap,

        "heterogeneity":
            heterogeneity,

        "kruskal_wallis":
            kruskal,

        "leave_one_out":
            loo.to_dict(
                orient="records"
            ),

        "interpretation": {

            "main":
                (
                    "A-before-B generality is supported "
                    "only when positive precedence persists "
                    "across sufficiently represented expression "
                    "categories using the same validated event "
                    "definitions."
                ),

            "threshold_control":
                (
                    "A and C thresholds are fixed to the "
                    "validated early-warning pipeline and "
                    "are not re-estimated per expression."
                ),

            "heterogeneity":
                (
                    "Expression-specific differences can be "
                    "scientifically meaningful because facial "
                    "expression geometry may change the "
                    "viewpoint at which representation instability "
                    "appears."
                ),

            "small_groups":
                (
                    f"Expressions with fewer than "
                    f"{MIN_PRIMARY_N} sequences are exploratory "
                    "and excluded from the primary generality claim."
                ),

            "causality":
                (
                    "The analysis tests statistical/temporal "
                    "precedence, not causal direction."
                ),
        },
    }

    path = (
        OUT
        / "cross_expression_report.json"
    )

    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


# ======================================================================
# README
# ======================================================================

def write_readme():

    text = f"""
# Cross-expression analysis

## Scientific question

Does the early-warning phenomenon generalize across
expression categories, or is it concentrated in a
small subset?

## Validated configuration

Frontal viewpoint:

    {FRONTAL}

Expected viewpoints:

    {EXPECTED_VIEWS}

A threshold:

    {A_THRESHOLD:.8f}

C threshold:

    {C_THRESHOLD:.8f}

Sustained viewpoints:

    {SUSTAINED}

Primary minimum expression size:

    {MIN_PRIMARY_N}

## A definition

A boundary is detected when:

    A_angular_distance_deg >= {A_THRESHOLD:.8f}

for {SUSTAINED} consecutive viewpoints.

## C definition

A boundary is detected when:

    C_margin <= {C_THRESHOLD:.8f}

for {SUSTAINED} consecutive viewpoints.

## B definition

A boundary is detected when:

    predicted_folder != true folder

for {SUSTAINED} consecutive viewpoints.

## Critical methodological point

Thresholds are FIXED.

They are not re-estimated for individual
expression categories.

This makes this analysis directly comparable
with the validated early-warning and horizon analyses.

## Primary analysis

Expression categories with at least
{MIN_PRIMARY_N} complete sequences are included
in the primary analysis.

Small groups remain visible in the complete
CSV files but are treated as exploratory.

## Interpretation

Positive A-before-B means:

    representation drift boundary
    occurs before
    prediction failure boundary.

This is evidence of temporal/statistical precedence.

It is not proof of causality.

## Output

sequence_events.csv
expression_coverage.csv
expression_summary_all.csv
expression_summary_primary.csv
expression_horizon_summary.csv
pooled_primary_summary.csv
bootstrap_expression_rates.csv
expression_heterogeneity_permutation.csv
kruskal_wallis.csv
leave_one_expression_out.csv
cross_expression_report.json

plots/expression_warning_rate_left.png
plots/expression_warning_rate_right.png
plots/expression_lead_left.png
plots/expression_lead_right.png
plots/expression_horizon_heatmap.png
"""

    (
        OUT
        / "README_cross_expression.md"
    ).write_text(
        text.strip(),
        encoding="utf-8",
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        + "#" * 70
    )

    print(
        "CROSS-EXPRESSION ANALYSIS"
    )

    print(
        "VALIDATED / CONSISTENT PIPELINE"
    )

    print(
        "#" * 70
    )

    print(
        "Project root:"
    )

    print(
        ROOT
    )

    print(
        "\nInput:"
    )

    print(
        INPUT
    )

    if not INPUT.exists():

        raise FileNotFoundError(
            INPUT
        )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "LOADING DATA"
    )

    print(
        "#" * 70
    )

    df = pd.read_csv(
        INPUT
    )

    c = get_columns(
        df
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns found: {len(df.columns)}"
    )

    print(
        "\nCOLUMN MAPPING"
    )

    for key, value in c.items():

        print(
            f"{key:16s}: {value}"
        )

    # ------------------------------------------------------------------
    # FIXED THRESHOLDS
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "VALIDATED THRESHOLDS — FIXED"
    )

    print(
        "#" * 70
    )

    print(
        f"A threshold : "
        f"{A_THRESHOLD:.8f}"
    )

    print(
        f"C threshold : "
        f"{C_THRESHOLD:.8f}"
    )

    print(
        f"Sustained viewpoints : "
        f"{SUSTAINED}"
    )

    print(
        "\nNO THRESHOLD ESTIMATION "
        "IS PERFORMED IN THIS SCRIPT."
    )

    # ------------------------------------------------------------------
    # BUILD EVENTS
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "BUILDING SEQUENCE EVENTS"
    )

    print(
        "#" * 70
    )

    events = build_events(
        df,
        c,
    )

    if events.empty:

        raise RuntimeError(
            "No complete 215-view sequences found."
        )

    events = add_distances(
        events
    )

    events.to_csv(
        OUT / "sequence_events.csv",
        index=False,
    )

    print(
        f"\nComplete sequences: "
        f"{len(events)}"
    )

    # ------------------------------------------------------------------
    # COVERAGE
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "EXPRESSION COVERAGE"
    )

    print(
        "#" * 70
    )

    coverage = (
        events.groupby(
            "expression"
        )
        .size()
        .rename(
            "n_sequences"
        )
        .reset_index()
    )

    coverage[
        "primary_eligible"
    ] = (
        coverage.n_sequences
        >= MIN_PRIMARY_N
    )

    coverage = coverage.sort_values(
        [
            "n_sequences",
            "expression",
        ],
        ascending=[
            False,
            True,
        ],
    )

    coverage.to_csv(
        OUT / "expression_coverage.csv",
        index=False,
    )

    print(
        f"Expression categories: "
        f"{len(coverage)}"
    )

    print(
        f"Primary eligible "
        f"(n >= {MIN_PRIMARY_N}): "
        f"{coverage.primary_eligible.sum()}"
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # EXPRESSION SUMMARY
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "EXPRESSION-LEVEL SUMMARY"
    )

    print(
        "#" * 70
    )

    summary = (
        build_expression_summary(
            events
        )
    )

    summary.to_csv(
        OUT
        / "expression_summary_all.csv",
        index=False,
    )

    primary = summary[
        summary.primary_eligible
    ].copy()

    primary.to_csv(
        OUT
        / "expression_summary_primary.csv",
        index=False,
    )

    for side in (
        "left",
        "right",
    ):

        x = primary[
            primary.side == side
        ]

        if x.empty:
            continue

        print(
            f"\n{side.upper()}"
        )

        print(
            f"Primary groups: "
            f"{len(x)}"
        )

        print(
            f"Mean A-before-B: "
            f"{x.A_before_B_percent.mean():.3f}%"
        )

        print(
            f"Median A-before-B: "
            f"{x.A_before_B_percent.median():.3f}%"
        )

        print(
            f"Range: "
            f"{x.A_before_B_percent.min():.3f}%"
            f" - "
            f"{x.A_before_B_percent.max():.3f}%"
        )

        print(
            f"Median expression lead: "
            f"{x.median_A_B_lead.median():.3f}°"
        )

    # ------------------------------------------------------------------
    # HORIZONS
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "CROSS-EXPRESSION HORIZON ANALYSIS"
    )

    print(
        "#" * 70
    )

    horizons = build_horizons(
        events
    )

    horizons.to_csv(
        OUT
        / "expression_horizon_summary.csv",
        index=False,
    )

    for side in (
        "left",
        "right",
    ):

        print(
            f"\n{side.upper()}"
        )

        x = horizons[
            (horizons.side == side)
            & horizons.primary_eligible
        ]

        for h in HORIZONS:

            z = x[
                x.horizon_degrees == h
            ]

            if z.empty:
                continue

            warning_count = int(
                z.warning_count.sum()
            )

            total = int(
                z.n.sum()
            )

            rate = (
                warning_count / total
                if total
                else np.nan
            )

            print(
                f"Horizon={h:3d}° | "
                f"warning={warning_count}/"
                f"{total} | "
                f"{100*rate:.3f}%"
            )

    # ------------------------------------------------------------------
    # POOLED PRIMARY
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "POOLED PRIMARY SUMMARY"
    )

    print(
        "#" * 70
    )

    pooled = pooled_primary(
        events
    )

    pooled.to_csv(
        OUT
        / "pooled_primary_summary.csv",
        index=False,
    )

    print(
        pooled.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # BOOTSTRAP
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "BOOTSTRAP"
    )

    print(
        "#" * 70
    )

    bootstrap_rows = []

    for side in (
        "left",
        "right",
    ):

        r = bootstrap_expression_rates(
            summary,
            side,
        )

        bootstrap_rows.append(
            r
        )

        print(
            f"{side.upper()} "
            f"mean={r['mean_rate']:.4f} "
            f"CI="
            f"[{r['ci_low']:.4f}, "
            f"{r['ci_high']:.4f}]"
        )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    bootstrap_df.to_csv(
        OUT
        / "bootstrap_expression_rates.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # HETEROGENEITY
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "EXPRESSION HETEROGENEITY"
    )

    print(
        "#" * 70
    )

    heterogeneity_rows = []

    for side in (
        "left",
        "right",
    ):

        r = heterogeneity_test(
            events,
            side,
        )

        heterogeneity_rows.append(
            r
        )

        print(
            f"{side.upper()} "
            f"observed range="
            f"{r['observed_range']:.4f} "
            f"null mean="
            f"{r['null_mean_range']:.4f} "
            f"p="
            f"{r['p_value']:.6g}"
        )

    heterogeneity_df = (
        pd.DataFrame(
            heterogeneity_rows
        )
    )

    heterogeneity_df.to_csv(
        OUT
        / "expression_heterogeneity_permutation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # KRUSKAL-WALLIS
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "KRUSKAL-WALLIS"
    )

    print(
        "#" * 70
    )

    kruskal_rows = []

    for side in (
        "left",
        "right",
    ):

        r = run_kruskal(
            events,
            side,
        )

        kruskal_rows.append(
            r
        )

        print(
            f"{side.upper()} "
            f"stat={r['statistic']} "
            f"p={r['p_value']}"
        )

    kruskal_df = pd.DataFrame(
        kruskal_rows
    )

    kruskal_df.to_csv(
        OUT
        / "kruskal_wallis.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # LEAVE ONE EXPRESSION OUT
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "LEAVE-ONE-EXPRESSION-OUT"
    )

    print(
        "#" * 70
    )

    loo = leave_one_out(
        events
    )

    loo.to_csv(
        OUT
        / "leave_one_expression_out.csv",
        index=False,
    )

    print(
        loo.head(
            20
        ).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # TOP / BOTTOM
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "TOP / BOTTOM EXPRESSIONS"
    )

    print(
        "#" * 70
    )

    for side in (
        "left",
        "right",
    ):

        x = primary[
            primary.side == side
        ].copy()

        print(
            f"\nTOP 10 — "
            f"{side.upper()}"
        )

        print(
            x.sort_values(
                "A_before_B_percent",
                ascending=False,
            )[
                [
                    "expression",
                    "n_sequences",
                    "A_before_B_percent",
                    "median_A_B_lead",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        print(
            f"\nBOTTOM 10 — "
            f"{side.upper()}"
        )

        print(
            x.sort_values(
                "A_before_B_percent",
                ascending=True,
            )[
                [
                    "expression",
                    "n_sequences",
                    "A_before_B_percent",
                    "median_A_B_lead",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "BUILDING PLOTS"
    )

    print(
        "#" * 70
    )

    plot_warning_rates(
        summary,
        "left",
        PLOTS
        / "expression_warning_rate_left.png",
    )

    plot_warning_rates(
        summary,
        "right",
        PLOTS
        / "expression_warning_rate_right.png",
    )

    plot_leads(
        summary,
        "left",
        PLOTS
        / "expression_lead_left.png",
    )

    plot_leads(
        summary,
        "right",
        PLOTS
        / "expression_lead_right.png",
    )

    plot_horizon_heatmap(
        horizons,
        PLOTS
        / "expression_horizon_heatmap.png",
    )

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "SAVING RESULTS"
    )

    print(
        "#" * 70
    )

    write_report(
        df,
        events,
        summary,
        horizons,
        pooled,
        bootstrap_rows,
        heterogeneity_rows,
        kruskal_rows,
        loo,
    )

    write_readme()

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print(
        "\n"
        + "#" * 70
    )

    print(
        "FINAL CROSS-EXPRESSION SUMMARY"
    )

    print(
        "#" * 70
    )

    print(
        f"Expressions analyzed: "
        f"{events.expression.nunique()}"
    )

    print(
        f"Primary eligible expressions: "
        f"{coverage.primary_eligible.sum()}"
    )

    for side in (
        "left",
        "right",
    ):

        x = primary[
            primary.side == side
        ]

        if x.empty:
            continue

        print(
            f"\n{side.upper()}"
        )

        print(
            f"Mean expression-level "
            f"A-before-B: "
            f"{x.A_before_B_percent.mean():.3f}%"
        )

        print(
            f"Median expression-level "
            f"A-before-B: "
            f"{x.A_before_B_percent.median():.3f}%"
        )

        print(
            f"Range: "
            f"{x.A_before_B_percent.min():.3f}%"
            f" - "
            f"{x.A_before_B_percent.max():.3f}%"
        )

        print(
            f"Median expression-level "
            f"lead: "
            f"{x.median_A_B_lead.median():.3f}°"
        )

        h = heterogeneity_df[
            heterogeneity_df.side == side
        ]

        if not h.empty:

            print(
                f"Heterogeneity permutation "
                f"p-value: "
                f"{h.iloc[0].p_value:.6g}"
            )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This version does NOT re-estimate "
        "A/C thresholds per expression."
    )

    print(
        "It uses the validated early-warning "
        "event definition."
    )

    print(
        "Small expression groups are exploratory "
        "and excluded from the primary generality claim."
    )

    print(
        "Positive A-before-B is evidence of "
        "temporal/statistical precedence, "
        "not causality."
    )

    print(
        "\nOutput directory:"
    )

    print(
        OUT
    )

    print(
        "\nDONE"
    )


# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    main()