# -*- coding: utf-8 -*-

"""
FER Reliability Benchmark
ROBUSTNESS / SENSITIVITY ANALYSIS

Purpose
-------
Test whether the main early-warning finding is robust to reasonable
changes in:

1. A threshold
2. C threshold
3. Sustained viewpoint requirement
4. A metric
5. Early-warning horizon
6. Left/right side
7. Bootstrap resampling
8. Permutation / shuffle null model

Main scientific question
------------------------
Does representation drift tend to precede prediction failure,
and does this conclusion survive reasonable changes in the
operational definitions?

IMPORTANT
---------
This script does NOT assume A < C < B.

A positive lead means:

    A_distance < B_distance

meaning representation drift is detected before prediction failure.

This is evidence of predictive precedence, NOT proof of causality.
"""

from __future__ import annotations

import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ANALYSIS_DIR = PROJECT_ROOT / "analysis"

INPUT_FILE = ANALYSIS_DIR / "4_analyze_embeddings_trajectory" / "per_view_metrics_multimetric.csv"

OUTPUT_DIR = ANALYSIS_DIR / "10_analyze_robustness_sensitivity"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# DEFAULT THRESHOLDS
# ------------------------------------------------------------

DEFAULT_A_THRESHOLD = 13.43702602
DEFAULT_C_THRESHOLD = 0.00237080

DEFAULT_SUSTAINED = 3


# ------------------------------------------------------------
# SENSITIVITY GRIDS
# ------------------------------------------------------------

A_THRESHOLD_FACTORS = [
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30,
]

C_THRESHOLD_FACTORS = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    2.00,
]

SUSTAINED_VALUES = [
    1,
    2,
    3,
    4,
    5,
]

HORIZONS = [
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
]

BOOTSTRAP_REPS = 1000

PERMUTATION_REPS = 2000

RANDOM_SEED = 1405


# ------------------------------------------------------------
# A METRICS
# ------------------------------------------------------------

A_METRICS = {
    "angular": "A_angular_distance_deg",
    "cosine": "A_cosine_distance",
    "euclidean": "A_euclidean_distance",
    "path": "A_cumulative_path_from_V107",
    "rate": "A_rate_per_degree",
    "curvature": "A_curvature",
    "instability": "A_trajectory_instability",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def banner(text: str):
    print()
    print("#" * 70)
    print(text)
    print("#" * 70)


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def bootstrap_ci(values, statistic=np.median, reps=1000, seed=1405):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    stats = []

    for _ in range(reps):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(statistic(sample))

    return (
        float(np.percentile(stats, 2.5)),
        float(np.percentile(stats, 97.5)),
    )


def permutation_pvalue(
    values,
    null_value=0.0,
    reps=2000,
    seed=1405,
):
    """
    Sign-flip permutation test.

    H0:
        median difference = 0

    We randomly flip signs of paired differences.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    observed = np.median(values)

    rng = np.random.default_rng(seed)

    count = 0

    for _ in range(reps):

        signs = rng.choice(
            [-1.0, 1.0],
            size=len(values),
        )

        permuted = np.median(values * signs)

        if abs(permuted) >= abs(observed):
            count += 1

    return (count + 1) / (reps + 1)


def proportion_ci(k, n, reps=10000, seed=1405):
    """
    Bootstrap CI for a proportion.
    """

    if n == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    p = k / n

    simulated = rng.binomial(
        n=n,
        p=p,
        size=reps,
    ) / n

    return (
        float(np.percentile(simulated, 2.5)),
        float(np.percentile(simulated, 97.5)),
    )


def normalize_viewpoint(df):
    df = df.copy()

    df["viewpoint"] = pd.to_numeric(
        df["viewpoint"],
        errors="coerce",
    )

    df = df.dropna(subset=["viewpoint"])

    df["viewpoint"] = df["viewpoint"].astype(float)

    return df


# ============================================================
# LOAD DATA
# ============================================================

banner("ROBUSTNESS / SENSITIVITY ANALYSIS")

print("Project root:")
print(PROJECT_ROOT)

print()
print("Analysis directory:")
print(ANALYSIS_DIR)

print()
print("Input:")
print(INPUT_FILE)


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)

print()
print("Rows loaded:", len(df))
print("Columns found:", len(df.columns))


# ============================================================
# COLUMN MAPPING
# ============================================================

REQUIRED = [
    "expression",
    "folder",
    "viewpoint",
    "A_angular_distance_deg",
    "C_margin",
    "B_predicted_folder",
]

missing = [
    c for c in REQUIRED
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


for col in [
    "viewpoint",
    "A_angular_distance_deg",
    "C_margin",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


print()
print("COLUMN MAPPING")
print("-" * 70)

print(
    "expression      : expression"
)

print(
    "folder          : folder"
)

print(
    "viewpoint       : viewpoint"
)

print(
    "A               : A_angular_distance_deg"
)

print(
    "C               : C_margin"
)

print(
    "B               : B_predicted_folder"
)


# ============================================================
# DATA PREPARATION
# ============================================================

df = normalize_viewpoint(df)

df["side"] = np.where(
    df["viewpoint"] < 107,
    "left",
    "right",
)

# ------------------------------------------------------------
# Distance from frontal viewpoint
# ------------------------------------------------------------

df["view_distance"] = np.abs(
    df["viewpoint"] - 107
)


# ============================================================
# EXPRESSION GROUPING
# ============================================================

group_cols = [
    "expression",
    "folder",
]


grouped = list(
    df.groupby(
        group_cols,
        sort=False,
    )
)


print()
print("Total expression sequences:", len(grouped))


# ============================================================
# EVENT DETECTION
# ============================================================

def first_sustained_viewpoint(
    viewpoints,
    condition,
    sustained=3,
):
    """
    Find first viewpoint where condition remains true
    for 'sustained' consecutive viewpoints.

    Returns viewpoint or NaN.
    """

    viewpoints = np.asarray(viewpoints)
    condition = np.asarray(condition, dtype=bool)

    if len(viewpoints) == 0:
        return np.nan

    order = np.argsort(viewpoints)

    viewpoints = viewpoints[order]
    condition = condition[order]

    if sustained <= 1:
        idx = np.where(condition)[0]

        if len(idx) == 0:
            return np.nan

        return float(viewpoints[idx[0]])

    for i in range(
        0,
        len(viewpoints) - sustained + 1,
    ):

        window_v = viewpoints[
            i:i + sustained
        ]

        window_c = condition[
            i:i + sustained
        ]

        # Require consecutive integer viewpoints
        consecutive = np.all(
            np.diff(window_v) == 1
        )

        if (
            consecutive
            and np.all(window_c)
        ):
            return float(window_v[0])

    return np.nan


# ============================================================
# B FAILURE DETECTION
# ============================================================

def detect_b_failure(
    group,
):
    """
    Detect first viewpoint where predicted folder
    differs from the expression's true folder.

    Returns NaN if no failure.
    """

    group = group.sort_values(
        "viewpoint"
    )

    true_folder = str(
        group["folder"].iloc[0]
    )

    prediction = (
        group["B_predicted_folder"]
        .astype(str)
    )

    failure = prediction != true_folder

    idx = np.where(
        failure.to_numpy()
    )[0]

    if len(idx) == 0:
        return np.nan

    return float(
        group.iloc[idx[0]]["viewpoint"]
    )


# ============================================================
# BUILD EVENTS
# ============================================================

def build_events(
    a_metric_col,
    a_threshold,
    c_threshold,
    sustained,
):
    rows = []

    for counter, (
        key,
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):

        expression, folder = key

        group = group.sort_values(
            "viewpoint"
        ).copy()

        # ----------------------------------------------------
        # A
        # ----------------------------------------------------

        A_condition = (
            group[a_metric_col]
            >= a_threshold
        )

        A_view = first_sustained_viewpoint(
            group["viewpoint"].to_numpy(),
            A_condition.to_numpy(),
            sustained=sustained,
        )

        # ----------------------------------------------------
        # C
        #
        # Low confidence:
        # C_margin <= threshold
        # ----------------------------------------------------

        C_condition = (
            group["C_margin"]
            <= c_threshold
        )

        C_view = first_sustained_viewpoint(
            group["viewpoint"].to_numpy(),
            C_condition.to_numpy(),
            sustained=sustained,
        )

        # ----------------------------------------------------
        # B
        # ----------------------------------------------------

        B_view = detect_b_failure(group)

        rows.append(
            {
                "expression": expression,
                "folder": folder,
                "A_view": A_view,
                "C_view": C_view,
                "B_view": B_view,
            }
        )

        if counter <= 5:
            print(
                f"Processed {counter}/{len(grouped)}"
            )

        elif counter in [
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            len(grouped),
        ]:
            print(
                f"Processed {counter}/{len(grouped)}"
            )

    events = pd.DataFrame(rows)

    return events


# ============================================================
# EVENT STATISTICS
# ============================================================

def event_statistics(events):
    results = []

    for side in ["left", "right"]:

        # The current event table does not directly contain
        # side-specific boundaries. We derive side events
        # using signed distance from frontal viewpoint.

        pass

    return pd.DataFrame(results)


# ============================================================
# BUILD SIDE-SPECIFIC EVENTS
# ============================================================

def build_side_events(
    a_metric_col,
    a_threshold,
    c_threshold,
    sustained,
):
    rows = []

    for key, group in grouped:

        expression, folder = key

        group = group.sort_values(
            "viewpoint"
        ).copy()

        for side in ["left", "right"]:

            if side == "left":

                side_group = group[
                    group["viewpoint"] <= 107
                ].copy()

                # analyze from frontal outward
                side_group = side_group.sort_values(
                    "viewpoint",
                    ascending=False,
                )

            else:

                side_group = group[
                    group["viewpoint"] >= 107
                ].copy()

                # analyze from frontal outward
                side_group = side_group.sort_values(
                    "viewpoint",
                    ascending=True,
                )

            if len(side_group) == 0:
                continue

            # ------------------------------------------------
            # Distance from frontal point
            # ------------------------------------------------

            side_group["distance"] = (
                np.abs(
                    side_group["viewpoint"]
                    - 107
                )
            )

            # ------------------------------------------------
            # A event
            # ------------------------------------------------

            A_condition = (
                side_group[a_metric_col]
                >= a_threshold
            ).to_numpy()

            C_condition = (
                side_group["C_margin"]
                <= c_threshold
            ).to_numpy()

            viewpoints = (
                side_group["viewpoint"]
                .to_numpy()
            )

            distances = (
                side_group["distance"]
                .to_numpy()
            )

            # ------------------------------------------------
            # Find first sustained event in outward order
            # ------------------------------------------------

            def first_distance(
                condition,
            ):

                if sustained <= 1:

                    idx = np.where(
                        condition
                    )[0]

                    if len(idx) == 0:
                        return np.nan

                    return float(
                        distances[idx[0]]
                    )

                for i in range(
                    0,
                    len(condition)
                    - sustained
                    + 1,
                ):

                    window = condition[
                        i:i+sustained
                    ]

                    if np.all(window):

                        return float(
                            distances[i]
                        )

                return np.nan

            A_distance = first_distance(
                A_condition
            )

            C_distance = first_distance(
                C_condition
            )

            # ------------------------------------------------
            # B failure
            # ------------------------------------------------

            true_folder = str(folder)

            predictions = (
                side_group[
                    "B_predicted_folder"
                ].astype(str)
            )

            failure = (
                predictions
                != true_folder
            ).to_numpy()

            B_distance = np.nan

            idx = np.where(
                failure
            )[0]

            if len(idx) > 0:

                B_distance = float(
                    distances[idx[0]]
                )

            rows.append(
                {
                    "expression": expression,
                    "folder": folder,
                    "side": side,
                    "A_distance": A_distance,
                    "C_distance": C_distance,
                    "B_distance": B_distance,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# BOUNDARY PRECEDENCE
# ============================================================

def precedence_summary(side_events):

    output = []

    for side in ["left", "right"]:

        x = side_events[
            side_events["side"] == side
        ].copy()

        x = x[
            x["A_distance"].notna()
            & x["B_distance"].notna()
        ]

        if len(x) == 0:

            output.append(
                {
                    "side": side,
                    "n": 0,
                    "A_before_B": np.nan,
                    "median_A": np.nan,
                    "median_B": np.nan,
                    "median_lead": np.nan,
                }
            )

            continue

        lead = (
            x["B_distance"]
            - x["A_distance"]
        )

        output.append(
            {
                "side": side,
                "n": len(x),
                "A_before_B": float(
                    (lead > 0).mean()
                ),
                "median_A": float(
                    x["A_distance"].median()
                ),
                "median_B": float(
                    x["B_distance"].median()
                ),
                "median_lead": float(
                    lead.median()
                ),
            }
        )

    return pd.DataFrame(output)


# ============================================================
# HORIZON SUMMARY
# ============================================================

def horizon_summary(
    side_events,
    horizon,
):
    rows = []

    for side in ["left", "right"]:

        x = side_events[
            side_events["side"] == side
        ].copy()

        x = x[
            x["A_distance"].notna()
            & x["B_distance"].notna()
        ]

        if len(x) == 0:

            rows.append(
                {
                    "side": side,
                    "horizon": horizon,
                    "n": 0,
                    "warning_count": 0,
                    "warning_rate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )

            continue

        lead = (
            x["B_distance"]
            - x["A_distance"]
        )

        warning = lead >= horizon

        count = int(
            warning.sum()
        )

        rate = (
            count / len(x)
        )

        ci_low, ci_high = proportion_ci(
            count,
            len(x),
        )

        rows.append(
            {
                "side": side,
                "horizon": horizon,
                "n": len(x),
                "warning_count": count,
                "warning_rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# BOOTSTRAP PRECEDENCE
# ============================================================

def bootstrap_precedence(
    side_events,
    reps=1000,
):
    rows = []

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    for side in ["left", "right"]:

        x = side_events[
            side_events["side"] == side
        ].copy()

        x = x[
            x["A_distance"].notna()
            & x["B_distance"].notna()
        ]

        if len(x) == 0:
            continue

        lead = (
            x["B_distance"]
            - x["A_distance"]
        ).to_numpy()

        observed = float(
            (lead > 0).mean()
        )

        stats = []

        for _ in range(reps):

            sample = rng.choice(
                lead,
                size=len(lead),
                replace=True,
            )

            stats.append(
                (sample > 0).mean()
            )

        rows.append(
            {
                "side": side,
                "n": len(lead),
                "observed_rate": observed,
                "ci_low": float(
                    np.percentile(
                        stats,
                        2.5,
                    )
                ),
                "ci_high": float(
                    np.percentile(
                        stats,
                        97.5,
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_precedence(
    side_events,
    reps=2000,
):
    """
    Null model:

    randomly swap A and B labels within each expression.

    Under H0 there should be no systematic precedence.
    """

    rows = []

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    for side in ["left", "right"]:

        x = side_events[
            side_events["side"] == side
        ].copy()

        x = x[
            x["A_distance"].notna()
            & x["B_distance"].notna()
        ]

        if len(x) == 0:
            continue

        A = x[
            "A_distance"
        ].to_numpy()

        B = x[
            "B_distance"
        ].to_numpy()

        observed = float(
            (B > A).mean()
        )

        null_rates = []

        for _ in range(reps):

            swap = rng.random(
                len(A)
            ) < 0.5

            perm_A = np.where(
                swap,
                B,
                A,
            )

            perm_B = np.where(
                swap,
                A,
                B,
            )

            rate = (
                perm_B > perm_A
            ).mean()

            null_rates.append(rate)

        null_rates = np.asarray(
            null_rates
        )

        p = (
            np.sum(
                null_rates >= observed
            ) + 1
        ) / (
            len(null_rates) + 1
        )

        rows.append(
            {
                "side": side,
                "n": len(A),
                "observed": observed,
                "null_mean": float(
                    null_rates.mean()
                ),
                "p_value": float(p),
                "permutation_reps": reps,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ROBUSTNESS EXPERIMENT
# ============================================================

all_sensitivity = []

all_precedence = []

all_horizons = []

all_bootstrap = []

all_permutations = []


# ============================================================
# 1. A THRESHOLD SENSITIVITY
# ============================================================

banner(
    "1. A-THRESHOLD SENSITIVITY"
)

for factor in A_THRESHOLD_FACTORS:

    threshold = (
        DEFAULT_A_THRESHOLD
        * factor
    )

    print(
        f"A factor={factor:.2f} "
        f"threshold={threshold:.6f}"
    )

    side_events = build_side_events(
        "A_angular_distance_deg",
        threshold,
        DEFAULT_C_THRESHOLD,
        DEFAULT_SUSTAINED,
    )

    summary = precedence_summary(
        side_events
    )

    for _, row in summary.iterrows():

        all_sensitivity.append(
            {
                "experiment": "A_threshold",
                "parameter": factor,
                "parameter_value": threshold,
                "side": row["side"],
                "n": row["n"],
                "A_before_B": row[
                    "A_before_B"
                ],
                "median_A": row[
                    "median_A"
                ],
                "median_B": row[
                    "median_B"
                ],
                "median_lead": row[
                    "median_lead"
                ],
            }
        )


# ============================================================
# 2. C THRESHOLD SENSITIVITY
# ============================================================

banner(
    "2. C-THRESHOLD SENSITIVITY"
)

for factor in C_THRESHOLD_FACTORS:

    threshold = (
        DEFAULT_C_THRESHOLD
        * factor
    )

    print(
        f"C factor={factor:.2f} "
        f"threshold={threshold:.8f}"
    )

    side_events = build_side_events(
        "A_angular_distance_deg",
        DEFAULT_A_THRESHOLD,
        threshold,
        DEFAULT_SUSTAINED,
    )

    summary = precedence_summary(
        side_events
    )

    for _, row in summary.iterrows():

        all_sensitivity.append(
            {
                "experiment": "C_threshold",
                "parameter": factor,
                "parameter_value": threshold,
                "side": row["side"],
                "n": row["n"],
                "A_before_B": row[
                    "A_before_B"
                ],
                "median_A": row[
                    "median_A"
                ],
                "median_B": row[
                    "median_B"
                ],
                "median_lead": row[
                    "median_lead"
                ],
            }
        )


# ============================================================
# 3. SUSTAINED VIEWPOINT SENSITIVITY
# ============================================================

banner(
    "3. SUSTAINED-VIEWPOINT SENSITIVITY"
)

for sustained in SUSTAINED_VALUES:

    print(
        f"Sustained viewpoints={sustained}"
    )

    side_events = build_side_events(
        "A_angular_distance_deg",
        DEFAULT_A_THRESHOLD,
        DEFAULT_C_THRESHOLD,
        sustained,
    )

    summary = precedence_summary(
        side_events
    )

    for _, row in summary.iterrows():

        all_sensitivity.append(
            {
                "experiment": "sustained",
                "parameter": sustained,
                "parameter_value": sustained,
                "side": row["side"],
                "n": row["n"],
                "A_before_B": row[
                    "A_before_B"
                ],
                "median_A": row[
                    "median_A"
                ],
                "median_B": row[
                    "median_B"
                ],
                "median_lead": row[
                    "median_lead"
                ],
            }
        )


# ============================================================
# 4. A METRIC SENSITIVITY
# ============================================================

banner(
    "4. A-METRIC SENSITIVITY"
)

for metric_name, metric_col in A_METRICS.items():

    if metric_col not in df.columns:

        print(
            f"SKIP {metric_name}: "
            f"{metric_col} not found"
        )

        continue

    values = pd.to_numeric(
        df[metric_col],
        errors="coerce",
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:

        print(
            f"SKIP {metric_name}: no finite values"
        )

        continue

    # --------------------------------------------------------
    # For alternative A metrics we use a robust
    # 90th percentile threshold.
    # --------------------------------------------------------

    threshold = float(
        np.percentile(
            finite,
            90,
        )
    )

    print(
        f"{metric_name}: "
        f"threshold={threshold:.8f}"
    )

    side_events = build_side_events(
        metric_col,
        threshold,
        DEFAULT_C_THRESHOLD,
        DEFAULT_SUSTAINED,
    )

    summary = precedence_summary(
        side_events
    )

    for _, row in summary.iterrows():

        all_sensitivity.append(
            {
                "experiment": "A_metric",
                "parameter": metric_name,
                "parameter_value": threshold,
                "side": row["side"],
                "n": row["n"],
                "A_before_B": row[
                    "A_before_B"
                ],
                "median_A": row[
                    "median_A"
                ],
                "median_B": row[
                    "median_B"
                ],
                "median_lead": row[
                    "median_lead"
                ],
            }
        )


# ============================================================
# 5. BASELINE EVENTS
# ============================================================

banner(
    "5. BASELINE ROBUSTNESS"
)

baseline_events = build_side_events(
    "A_angular_distance_deg",
    DEFAULT_A_THRESHOLD,
    DEFAULT_C_THRESHOLD,
    DEFAULT_SUSTAINED,
)

baseline_precedence = precedence_summary(
    baseline_events
)

print(
    baseline_precedence.to_string(
        index=False
    )
)


# ============================================================
# 6. HORIZON SENSITIVITY
# ============================================================

banner(
    "6. HORIZON SENSITIVITY"
)

for horizon in HORIZONS:

    result = horizon_summary(
        baseline_events,
        horizon,
    )

    all_horizons.append(
        result
    )

horizon_df = pd.concat(
    all_horizons,
    ignore_index=True,
)

print(
    horizon_df.to_string(
        index=False
    )
)


# ============================================================
# 7. BOOTSTRAP
# ============================================================

banner(
    "7. BOOTSTRAP STABILITY"
)

bootstrap_df = bootstrap_precedence(
    baseline_events,
    reps=BOOTSTRAP_REPS,
)

print(
    bootstrap_df.to_string(
        index=False
    )
)


# ============================================================
# 8. PERMUTATION
# ============================================================

banner(
    "8. PERMUTATION ROBUSTNESS"
)

permutation_df = permutation_precedence(
    baseline_events,
    reps=PERMUTATION_REPS,
)

print(
    permutation_df.to_string(
        index=False
    )
)


# ============================================================
# 9. COMBINED SENSITIVITY TABLE
# ============================================================

sensitivity_df = pd.DataFrame(
    all_sensitivity
)


# ============================================================
# 10. ROBUSTNESS SCORE
# ============================================================

banner(
    "10. ROBUSTNESS SCORE"
)

robust_rows = []

for side in ["left", "right"]:

    baseline = baseline_precedence[
        baseline_precedence["side"]
        == side
    ]

    if len(baseline) == 0:
        continue

    baseline_rate = float(
        baseline.iloc[0]["A_before_B"]
    )

    baseline_lead = float(
        baseline.iloc[0]["median_lead"]
    )

    subset = sensitivity_df[
        sensitivity_df["side"] == side
    ].copy()

    subset = subset[
        subset["A_before_B"].notna()
    ]

    if len(subset) == 0:
        continue

    # --------------------------------------------------------
    # Stability criteria
    #
    # A-before-B >= 80%
    # --------------------------------------------------------

    stable_80 = (
        subset["A_before_B"]
        >= 0.80
    ).mean()

    stable_70 = (
        subset["A_before_B"]
        >= 0.70
    ).mean()

    stable_positive = (
        subset["median_lead"]
        > 0
    ).mean()

    robust_score = np.mean(
        [
            stable_80,
            stable_70,
            stable_positive,
        ]
    )

    robust_rows.append(
        {
            "side": side,
            "baseline_A_before_B": baseline_rate,
            "baseline_median_lead": baseline_lead,
            "fraction_A_before_B_ge_80": stable_80,
            "fraction_A_before_B_ge_70": stable_70,
            "fraction_positive_median_lead": stable_positive,
            "overall_robustness_score": robust_score,
        }
    )

robustness_df = pd.DataFrame(
    robust_rows
)

print(
    robustness_df.to_string(
        index=False
    )
)


# ============================================================
# 11. LEFT / RIGHT CONSISTENCY
# ============================================================

banner(
    "11. LEFT / RIGHT CONSISTENCY"
)

left = baseline_precedence[
    baseline_precedence["side"]
    == "left"
]

right = baseline_precedence[
    baseline_precedence["side"]
    == "right"
]

if (
    len(left) > 0
    and len(right) > 0
):

    left_rate = float(
        left.iloc[0]["A_before_B"]
    )

    right_rate = float(
        right.iloc[0]["A_before_B"]
    )

    left_lead = float(
        left.iloc[0]["median_lead"]
    )

    right_lead = float(
        right.iloc[0]["median_lead"]
    )

    print(
        f"LEFT  A-before-B = "
        f"{left_rate * 100:.3f}%"
    )

    print(
        f"RIGHT A-before-B = "
        f"{right_rate * 100:.3f}%"
    )

    print(
        f"Difference = "
        f"{abs(left_rate-right_rate)*100:.3f} percentage points"
    )

    print(
        f"LEFT median lead = "
        f"{left_lead:.3f}°"
    )

    print(
        f"RIGHT median lead = "
        f"{right_lead:.3f}°"
    )


# ============================================================
# 12. SAVE CSV
# ============================================================

banner(
    "SAVING RESULTS"
)

sensitivity_file = (
    OUTPUT_DIR
    / "robustness_sensitivity_results.csv"
)

horizon_file = (
    OUTPUT_DIR
    / "robustness_horizon_results.csv"
)

bootstrap_file = (
    OUTPUT_DIR
    / "robustness_bootstrap.csv"
)

permutation_file = (
    OUTPUT_DIR
    / "robustness_permutation.csv"
)

robustness_file = (
    OUTPUT_DIR
    / "robustness_score.csv"
)


sensitivity_df.to_csv(
    sensitivity_file,
    index=False,
)

horizon_df.to_csv(
    horizon_file,
    index=False,
)

bootstrap_df.to_csv(
    bootstrap_file,
    index=False,
)

permutation_df.to_csv(
    permutation_file,
    index=False,
)

robustness_df.to_csv(
    robustness_file,
    index=False,
)


# ============================================================
# 13. PLOT A THRESHOLD ROBUSTNESS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for side in ["left", "right"]:

    x = sensitivity_df[
        (
            sensitivity_df[
                "experiment"
            ]
            == "A_threshold"
        )
        & (
            sensitivity_df["side"]
            == side
        )
    ].copy()

    if len(x) == 0:
        continue

    x = x.sort_values(
        "parameter_value"
    )

    plt.plot(
        x["parameter"],
        x["A_before_B"] * 100,
        marker="o",
        label=side,
    )

plt.axhline(
    80,
    linestyle="--",
)

plt.xlabel(
    "A threshold multiplier"
)

plt.ylabel(
    "A-before-B (%)"
)

plt.title(
    "Robustness to A-threshold"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "robustness_A_threshold.png",
    dpi=200,
)

plt.close()


# ============================================================
# 14. PLOT SUSTAINED SENSITIVITY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for side in ["left", "right"]:

    x = sensitivity_df[
        (
            sensitivity_df[
                "experiment"
            ]
            == "sustained"
        )
        & (
            sensitivity_df["side"]
            == side
        )
    ].copy()

    if len(x) == 0:
        continue

    x = x.sort_values(
        "parameter_value"
    )

    plt.plot(
        x["parameter_value"],
        x["A_before_B"] * 100,
        marker="o",
        label=side,
    )

plt.axhline(
    80,
    linestyle="--",
)

plt.xlabel(
    "Sustained viewpoints"
)

plt.ylabel(
    "A-before-B (%)"
)

plt.title(
    "Robustness to Sustained-Viewpoint Requirement"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "robustness_sustained.png",
    dpi=200,
)

plt.close()


# ============================================================
# 15. PLOT HORIZON
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for side in ["left", "right"]:

    x = horizon_df[
        horizon_df["side"]
        == side
    ].copy()

    if len(x) == 0:
        continue

    x = x.sort_values(
        "horizon"
    )

    plt.plot(
        x["horizon"],
        x["warning_rate"] * 100,
        marker="o",
        label=side,
    )

plt.axhline(
    50,
    linestyle="--",
)

plt.xlabel(
    "Warning horizon (degrees)"
)

plt.ylabel(
    "Warning rate (%)"
)

plt.title(
    "Early-warning horizon robustness"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "robustness_horizon.png",
    dpi=200,
)

plt.close()


# ============================================================
# 16. PLOT A METRIC ROBUSTNESS
# ============================================================

metric_df = sensitivity_df[
    sensitivity_df["experiment"]
    == "A_metric"
].copy()

if len(metric_df) > 0:

    pivot = metric_df.pivot_table(
        index="parameter",
        columns="side",
        values="A_before_B",
        aggfunc="mean",
    )

    plt.figure(
        figsize=(11, 7)
    )

    pivot.plot(
        kind="bar",
        figsize=(11, 7),
    )

    plt.axhline(
        0.80,
        linestyle="--",
    )

    plt.ylabel(
        "A-before-B"
    )

    plt.xlabel(
        "A metric"
    )

    plt.title(
        "Cross-metric robustness"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "robustness_A_metrics.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# 17. JSON REPORT
# ============================================================

report = {
    "project_root": str(
        PROJECT_ROOT
    ),

    "input_file": str(
        INPUT_FILE
    ),

    "rows_loaded": int(
        len(df)
    ),

    "expression_sequences": int(
        len(grouped)
    ),

    "baseline": {
        "A_threshold": DEFAULT_A_THRESHOLD,
        "C_threshold": DEFAULT_C_THRESHOLD,
        "sustained_viewpoints": DEFAULT_SUSTAINED,
        "A_metric": "A_angular_distance_deg",
    },

    "sensitivity": {
        "A_threshold_factors":
            A_THRESHOLD_FACTORS,

        "C_threshold_factors":
            C_THRESHOLD_FACTORS,

        "sustained_values":
            SUSTAINED_VALUES,

        "horizons":
            HORIZONS,

        "A_metrics":
            list(A_METRICS.keys()),
    },

    "bootstrap_repetitions":
        BOOTSTRAP_REPS,

    "permutation_repetitions":
        PERMUTATION_REPS,

    "baseline_precedence":
        baseline_precedence.to_dict(
            orient="records"
        ),

    "robustness_score":
        robustness_df.to_dict(
            orient="records"
        ),

    "interpretation": {
        "positive_lead":
            "A representation drift boundary occurs before B prediction failure.",

        "causality_warning":
            "Robust precedence is not proof of causality.",

        "purpose":
            "Determine whether the observed early-warning pattern survives reasonable operational changes.",
    },
}


json_file = (
    OUTPUT_DIR
    / "robustness_sensitivity_report.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 18. README
# ============================================================

readme = f"""
# Robustness / Sensitivity Analysis

## Purpose

This analysis tests whether the main early-warning result is
sensitive to reasonable changes in the operational definitions.

The main question is:

> Does representation drift tend to precede prediction failure,
> and does this conclusion survive reasonable analytical choices?

## Baseline

A threshold:
{DEFAULT_A_THRESHOLD}

C threshold:
{DEFAULT_C_THRESHOLD}

Sustained viewpoints:
{DEFAULT_SUSTAINED}

Primary A metric:
A_angular_distance_deg

## Sensitivity tests

### 1. A threshold

The A threshold is multiplied by:

{A_THRESHOLD_FACTORS}

### 2. C threshold

The C threshold is multiplied by:

{C_THRESHOLD_FACTORS}

### 3. Sustained viewpoints

Tested values:

{SUSTAINED_VALUES}

### 4. A metrics

Tested metrics:

{list(A_METRICS.keys())}

### 5. Early-warning horizons

Tested horizons:

{HORIZONS}

### 6. Bootstrap

Bootstrap repetitions:

{BOOTSTRAP_REPS}

### 7. Permutation

Permutation repetitions:

{PERMUTATION_REPS}

## Interpretation

A result is considered more robust when:

1. A-before-B remains high across reasonable thresholds.
2. Median A-before-B lead remains positive.
3. Left and right sides show similar qualitative behavior.
4. Bootstrap confidence intervals remain away from zero.
5. Permutation tests remain significant.
6. Results do not depend on one specific A metric.

## Important

Robustness does not establish causality.

It establishes that the observed statistical pattern is
not easily explained by one arbitrary choice of threshold,
sustained duration, metric, or side.

## Main output

See:

robustness_sensitivity_report.json

robustness_sensitivity_results.csv

robustness_horizon_results.csv

robustness_bootstrap.csv

robustness_permutation.csv

robustness_score.csv
"""


readme_file = (
    OUTPUT_DIR
    / "README_robustness_sensitivity.md"
)

with open(
    readme_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(readme)


# ============================================================
# FINAL SUMMARY
# ============================================================

banner(
    "FINAL ROBUSTNESS SUMMARY"
)

for _, row in baseline_precedence.iterrows():

    side = row["side"]

    print(
        f"{side.upper():5s} "
        f"A-before-B="
        f"{row['A_before_B'] * 100:.3f}% "
        f"median lead="
        f"{row['median_lead']:.1f}°"
    )


if len(robustness_df) > 0:

    print()
    print(
        "ROBUSTNESS SCORES"
    )

    print(
        robustness_df.to_string(
            index=False
        )
    )


print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Output files:"
)

for path in [
    sensitivity_file,
    horizon_file,
    bootstrap_file,
    permutation_file,
    robustness_file,
    json_file,
    readme_file,
]:

    print(path)


print()
print(
    "IMPORTANT:"
)

print(
    "This analysis tests robustness of the observed "
    "early-warning pattern."
)

print(
    "A positive A-before-B lead means representation "
    "drift precedes prediction failure."
)

print(
    "Robustness across thresholds and metrics strengthens "
    "the empirical claim."
)

print(
    "It does NOT establish causal direction."
)

print()
print(
    "DONE"
)