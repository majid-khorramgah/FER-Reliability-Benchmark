from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_VIEWPOINTS = 215
MIN_VIEWPOINT = 0
MAX_VIEWPOINT = 214
FRONTAL_VIEWPOINT = 107

BASELINE_HALF_WIDTH = 5
SUSTAINED_COUNT = 3

BOOTSTRAP_REPETITIONS = 500
RANDOM_SEED = 42

# Primary representation metric.
PRIMARY_A_METRIC = "A_angular_distance_deg"

# Existing thresholds from the previous analysis.
DEFAULT_THRESHOLDS = {
    "A_angular": 13.43702602,
    "A_cosine": 0.02737410,
    "A_euclidean": 0.23398377,
    "A_path": 0.41527239,
    "A_rate": 0.10537625,
    "A_curvature": 43.19991900,
    "A_instability": 0.16426415,
    "C_margin": 0.00237080,
}


# ============================================================
# PATHS
# ============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parent


ROOT = project_root()

ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR_OUTPUT = ROOT / "analysis" / "6_analyze_ordering_diagnostic"

METRICS_FILE = (
    ANALYSIS_DIR / "4_analyze_embeddings_trajectory" /
    "per_view_metrics_multimetric.csv"
)

TRAJECTORY_REPORT = (
    ANALYSIS_DIR_OUTPUT /
    "trajectory_analysis_report.json"
)

OUTPUT_EXPRESSION = (
    ANALYSIS_DIR_OUTPUT /
    "ordering_diagnostic_expression.csv"
)

OUTPUT_POPULATION = (
    ANALYSIS_DIR_OUTPUT /
    "ordering_diagnostic_population.csv"
)

OUTPUT_DIRECTIONAL = (
    ANALYSIS_DIR_OUTPUT /
    "ordering_diagnostic_directional.csv"
)

OUTPUT_BOOTSTRAP = (
    ANALYSIS_DIR_OUTPUT /
    "ordering_diagnostic_bootstrap.csv"
)

OUTPUT_REPORT = (
    ANALYSIS_DIR_OUTPUT /
    "ordering_diagnostic_report.json"
)

OUTPUT_README = (
    ANALYSIS_DIR_OUTPUT /
    "README_ordering_diagnostic.md"
)

PLOT_DIR = (
    ANALYSIS_DIR_OUTPUT /
    "ordering_diagnostic_plots"
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class BoundaryResult:
    boundary: float
    detected: bool
    baseline_value: float
    threshold: float
    direction: str
    metric: str
    reason: str


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("#" * 70)
    print(title)
    print("#" * 70)


def safe_float(value) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass

    return np.nan


def normalize_expression(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# LOAD THRESHOLDS
# ============================================================

def load_thresholds() -> Dict[str, float]:

    thresholds = dict(DEFAULT_THRESHOLDS)

    if not TRAJECTORY_REPORT.exists():
        print(
            "Trajectory report not found."
        )
        print(
            "Using default thresholds."
        )
        return thresholds

    try:

        with open(
            TRAJECTORY_REPORT,
            "r",
            encoding="utf-8"
        ) as f:
            report = json.load(f)

    except Exception as exc:

        print(
            f"Could not read trajectory report: {exc}"
        )

        return thresholds

    # Different versions of the previous script may
    # store thresholds under different keys.
    possible_maps = [
        report.get("thresholds"),
        report.get("robust_thresholds"),
        report.get("ROBUST_THRESHOLDS"),
    ]

    for candidate in possible_maps:

        if not isinstance(candidate, dict):
            continue

        for key, value in candidate.items():

            numeric = safe_float(value)

            if not np.isfinite(numeric):
                continue

            key_normalized = str(key)

            if key_normalized in thresholds:
                thresholds[key_normalized] = numeric

            elif key_normalized == "A_angular_threshold":
                thresholds["A_angular"] = numeric

            elif key_normalized == "C_margin_threshold":
                thresholds["C_margin"] = numeric

    return thresholds


# ============================================================
# LOAD METADATA
# ============================================================

def load_metrics() -> pd.DataFrame:

    if not METRICS_FILE.exists():

        raise FileNotFoundError(
            f"Metrics file not found:\n{METRICS_FILE}"
        )

    print(
        f"Loading:\n{METRICS_FILE}"
    )

    df = pd.read_csv(
        METRICS_FILE
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns found: {len(df.columns)}"
    )

    required = [
        "expression",
        "folder",
        "viewpoint",
        "A_angular_distance_deg",
        "C_margin",
        "B_predicted_folder",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    df["expression"] = (
        df["expression"]
        .map(normalize_expression)
    )

    df["folder"] = (
        df["folder"]
        .map(normalize_expression)
    )

    df["viewpoint"] = (
        pd.to_numeric(
            df["viewpoint"],
            errors="coerce"
        )
    )

    df["A_angular_distance_deg"] = (
        clean_numeric(
            df["A_angular_distance_deg"]
        )
    )

    df["C_margin"] = (
        clean_numeric(
            df["C_margin"]
        )
    )

    return df


# ============================================================
# COMPLETE SEQUENCES
# ============================================================

def find_complete_sequences(
    df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:

    sequences = {}

    grouped = df.groupby(
        "folder",
        sort=False
    )

    for folder, group in grouped:

        viewpoints = (
            pd.to_numeric(
                group["viewpoint"],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .unique()
        )

        expected = set(
            range(
                MIN_VIEWPOINT,
                MAX_VIEWPOINT + 1
            )
        )

        observed = set(viewpoints)

        if observed != expected:
            continue

        # Remove duplicate viewpoints.
        counts = (
            group["viewpoint"]
            .value_counts()
        )

        if (
            len(counts) != EXPECTED_VIEWPOINTS
            or counts.max() != 1
        ):
            continue

        group = (
            group
            .sort_values("viewpoint")
            .reset_index(drop=True)
        )

        if len(group) != EXPECTED_VIEWPOINTS:
            continue

        sequences[str(folder)] = group

    return sequences


# ============================================================
# BASELINE
# ============================================================

def baseline_value(
    sequence: pd.DataFrame,
    column: str
) -> float:

    baseline = sequence[
        sequence["viewpoint"]
        .between(
            FRONTAL_VIEWPOINT - BASELINE_HALF_WIDTH,
            FRONTAL_VIEWPOINT + BASELINE_HALF_WIDTH
        )
    ][column]

    baseline = (
        pd.to_numeric(
            baseline,
            errors="coerce"
        )
        .dropna()
    )

    if baseline.empty:
        return np.nan

    return float(
        np.nanmedian(
            baseline.to_numpy(
                dtype=float
            )
        )
    )


# ============================================================
# A BOUNDARY
# ============================================================

def detect_A_boundary(
    sequence: pd.DataFrame,
    direction: str,
    threshold: float
) -> BoundaryResult:

    metric = PRIMARY_A_METRIC

    values = (
        pd.to_numeric(
            sequence[metric],
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )

    viewpoints = (
        sequence["viewpoint"]
        .to_numpy(
            dtype=float
        )
    )

    baseline = baseline_value(
        sequence,
        metric
    )

    if not np.isfinite(baseline):

        return BoundaryResult(
            boundary=np.nan,
            detected=False,
            baseline_value=np.nan,
            threshold=threshold,
            direction=direction,
            metric=metric,
            reason="missing_baseline"
        )

    # We use threshold as the absolute robust boundary,
    # consistent with the previous analysis.
    condition = (
        values >= threshold
    )

    if direction == "left":

        # Search from frontal toward smaller viewpoints.
        indices = np.where(
            viewpoints <= FRONTAL_VIEWPOINT
        )[0]

        indices = indices[
            np.argsort(
                viewpoints[indices]
            )[::-1]
        ]

    else:

        # Search from frontal toward larger viewpoints.
        indices = np.where(
            viewpoints >= FRONTAL_VIEWPOINT
        )[0]

        indices = indices[
            np.argsort(
                viewpoints[indices]
            )
        ]

    consecutive = 0

    for idx in indices:

        value = values[idx]

        if not np.isfinite(value):

            consecutive = 0
            continue

        if condition[idx]:

            consecutive += 1

            if consecutive >= SUSTAINED_COUNT:

                # The first point of the sustained run
                # is the detected boundary.
                start_idx = (
                    indices[
                        np.where(
                            indices == idx
                        )[0][0]
                        - SUSTAINED_COUNT
                        + 1
                    ]
                )

                boundary = viewpoints[
                    start_idx
                ]

                return BoundaryResult(
                    boundary=float(boundary),
                    detected=True,
                    baseline_value=float(baseline),
                    threshold=float(threshold),
                    direction=direction,
                    metric=metric,
                    reason="sustained_threshold_crossing"
                )

        else:

            consecutive = 0

    return BoundaryResult(
        boundary=np.nan,
        detected=False,
        baseline_value=float(baseline),
        threshold=float(threshold),
        direction=direction,
        metric=metric,
        reason="no_sustained_crossing"
    )


# ============================================================
# C BOUNDARY
# ============================================================

def detect_C_boundary(
    sequence: pd.DataFrame,
    direction: str,
    threshold: float
) -> BoundaryResult:

    metric = "C_margin"

    values = (
        pd.to_numeric(
            sequence[metric],
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )

    viewpoints = (
        sequence["viewpoint"]
        .to_numpy(
            dtype=float
        )
    )

    baseline = baseline_value(
        sequence,
        metric
    )

    if not np.isfinite(baseline):

        return BoundaryResult(
            boundary=np.nan,
            detected=False,
            baseline_value=np.nan,
            threshold=threshold,
            direction=direction,
            metric=metric,
            reason="missing_baseline"
        )

    # C_margin becomes problematic when it gets small.
    condition = (
        values <= threshold
    )

    if direction == "left":

        indices = np.where(
            viewpoints <= FRONTAL_VIEWPOINT
        )[0]

        indices = indices[
            np.argsort(
                viewpoints[indices]
            )[::-1]
        ]

    else:

        indices = np.where(
            viewpoints >= FRONTAL_VIEWPOINT
        )[0]

        indices = indices[
            np.argsort(
                viewpoints[indices]
            )
        ]

    consecutive = 0

    for idx in indices:

        value = values[idx]

        if not np.isfinite(value):

            consecutive = 0
            continue

        if condition[idx]:

            consecutive += 1

            if consecutive >= SUSTAINED_COUNT:

                start_position = (
                    np.where(
                        indices == idx
                    )[0][0]
                    - SUSTAINED_COUNT
                    + 1
                )

                start_idx = indices[
                    start_position
                ]

                boundary = viewpoints[
                    start_idx
                ]

                return BoundaryResult(
                    boundary=float(boundary),
                    detected=True,
                    baseline_value=float(baseline),
                    threshold=float(threshold),
                    direction=direction,
                    metric=metric,
                    reason="sustained_threshold_crossing"
                )

        else:

            consecutive = 0

    return BoundaryResult(
        boundary=np.nan,
        detected=False,
        baseline_value=float(baseline),
        threshold=float(threshold),
        direction=direction,
        metric=metric,
        reason="no_sustained_crossing"
    )


# ============================================================
# B BOUNDARY
# ============================================================

def detect_B_boundary(
    sequence: pd.DataFrame,
    direction: str
) -> BoundaryResult:

    viewpoints = (
        sequence["viewpoint"]
        .to_numpy(
            dtype=float
        )
    )

    predictions = (
        sequence["B_predicted_folder"]
        .map(normalize_expression)
        .to_numpy()
    )

    true_folder = normalize_expression(
        sequence["folder"].iloc[0]
    )

    correct = (
        predictions == true_folder
    )

    if direction == "left":

        indices = np.where(
            viewpoints <= FRONTAL_VIEWPOINT
        )[0]

        indices = indices[
            np.argsort(
                viewpoints[indices]
            )[::-1]
        ]

    else:

        indices = np.where(
            viewpoints >= FRONTAL_VIEWPOINT
        )[0]

        indices = indices[
            np.argsort(
                viewpoints[indices]
            )
        ]

    consecutive = 0

    for idx in indices:

        value = correct[idx]

        if not isinstance(value, (bool, np.bool_)):
            value = bool(value)

        if not value:

            consecutive += 1

            if consecutive >= SUSTAINED_COUNT:

                start_position = (
                    np.where(
                        indices == idx
                    )[0][0]
                    - SUSTAINED_COUNT
                    + 1
                )

                start_idx = indices[
                    start_position
                ]

                boundary = viewpoints[
                    start_idx
                ]

                return BoundaryResult(
                    boundary=float(boundary),
                    detected=True,
                    baseline_value=np.nan,
                    threshold=np.nan,
                    direction=direction,
                    metric="B_prediction",
                    reason="sustained_prediction_failure"
                )

        else:

            consecutive = 0

    return BoundaryResult(
        boundary=np.nan,
        detected=False,
        baseline_value=np.nan,
        threshold=np.nan,
        direction=direction,
        metric="B_prediction",
        reason="no_sustained_prediction_failure"
    )


# ============================================================
# ORDERING
# ============================================================

def classify_ordering(
    a: float,
    c: float,
    b: float
) -> str:

    values = {
        "A": a,
        "C": c,
        "B": b,
    }

    if not all(
        np.isfinite(v)
        for v in values.values()
    ):
        missing = [
            key
            for key, value in values.items()
            if not np.isfinite(value)
        ]

        return (
            "missing_" +
            "_".join(missing)
        )

    ordered = sorted(
        values.items(),
        key=lambda x: x[1]
    )

    # Exact ties.
    if (
        abs(a - c) < 1e-9
        and abs(c - b) < 1e-9
    ):
        return "tie_all"

    if abs(a - c) < 1e-9:
        return "A=C<B" if a < b else "B<A=C"

    if abs(a - b) < 1e-9:
        return "A=B<C" if a < c else "C<A=B"

    if abs(c - b) < 1e-9:
        return "C=B<A" if c < a else "A<C=B"

    return (
        ordered[0][0]
        + "<"
        + ordered[1][0]
        + "<"
        + ordered[2][0]
    )


# ============================================================
# ONE EXPRESSION
# ============================================================

def analyze_expression(
    sequence: pd.DataFrame,
    thresholds: Dict[str, float]
) -> Dict:

    folder = normalize_expression(
        sequence["folder"].iloc[0]
    )

    expression = normalize_expression(
        sequence["expression"].iloc[0]
    )

    results = {
        "folder": folder,
        "expression": expression,
    }

    for direction in ("left", "right"):

        A = detect_A_boundary(
            sequence,
            direction,
            thresholds["A_angular"]
        )

        C = detect_C_boundary(
            sequence,
            direction,
            thresholds["C_margin"]
        )

        B = detect_B_boundary(
            sequence,
            direction
        )

        a = A.boundary
        c = C.boundary
        b = B.boundary

        ordering = classify_ordering(
            a,
            c,
            b
        )

        results[f"A_{direction}"] = a
        results[f"C_{direction}"] = c
        results[f"B_{direction}"] = b

        results[f"A_{direction}_detected"] = A.detected
        results[f"C_{direction}_detected"] = C.detected
        results[f"B_{direction}_detected"] = B.detected

        results[f"ordering_{direction}"] = ordering

        if (
            np.isfinite(a)
            and np.isfinite(c)
        ):
            results[
                f"A_to_C_gap_{direction}"
            ] = abs(c - a)
        else:
            results[
                f"A_to_C_gap_{direction}"
            ] = np.nan

        if (
            np.isfinite(c)
            and np.isfinite(b)
        ):
            results[
                f"C_to_B_gap_{direction}"
            ] = abs(b - c)
        else:
            results[
                f"C_to_B_gap_{direction}"
            ] = np.nan

        if (
            np.isfinite(a)
            and np.isfinite(b)
        ):
            results[
                f"A_to_B_gap_{direction}"
            ] = abs(b - a)
        else:
            results[
                f"A_to_B_gap_{direction}"
            ] = np.nan

        results[
            f"A_reason_{direction}"
        ] = A.reason

        results[
            f"C_reason_{direction}"
        ] = C.reason

        results[
            f"B_reason_{direction}"
        ] = B.reason

    return results


# ============================================================
# POPULATION SUMMARY
# ============================================================

def population_summary(
    expression_df: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    for side in ("left", "right"):

        column = f"ordering_{side}"

        counts = (
            expression_df[column]
            .value_counts(
                dropna=False
            )
        )

        total = len(
            expression_df
        )

        for ordering, count in counts.items():

            rows.append({
                "side": side,
                "ordering": ordering,
                "count": int(count),
                "proportion": (
                    float(count) /
                    float(total)
                    if total > 0
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


# ============================================================
# DETECTION DIAGNOSTICS
# ============================================================

def detection_diagnostics(
    expression_df: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    for side in ("left", "right"):

        total = len(
            expression_df
        )

        a_detected = int(
            expression_df[
                f"A_{side}_detected"
            ].sum()
        )

        c_detected = int(
            expression_df[
                f"C_{side}_detected"
            ].sum()
        )

        b_detected = int(
            expression_df[
                f"B_{side}_detected"
            ].sum()
        )

        all_three = int(
            (
                expression_df[
                    f"A_{side}_detected"
                ]
                &
                expression_df[
                    f"C_{side}_detected"
                ]
                &
                expression_df[
                    f"B_{side}_detected"
                ]
            ).sum()
        )

        rows.append({
            "side": side,
            "total_expressions": total,
            "A_detected": a_detected,
            "C_detected": c_detected,
            "B_detected": b_detected,
            "A_missing": total - a_detected,
            "C_missing": total - c_detected,
            "B_missing": total - b_detected,
            "all_three": all_three,
        })

    return pd.DataFrame(rows)


# ============================================================
# DIRECTIONAL SUMMARY
# ============================================================

def directional_summary(
    expression_df: pd.DataFrame
) -> pd.DataFrame:

    rows = []

    for side in ("left", "right"):

        a = pd.to_numeric(
            expression_df[f"A_{side}"],
            errors="coerce"
        )

        c = pd.to_numeric(
            expression_df[f"C_{side}"],
            errors="coerce"
        )

        b = pd.to_numeric(
            expression_df[f"B_{side}"],
            errors="coerce"
        )

        complete = (
            a.notna()
            & c.notna()
            & b.notna()
        )

        subset = expression_df[
            complete
        ].copy()

        if len(subset) == 0:
            continue

        aa = a[complete]
        cc = c[complete]
        bb = b[complete]

        rows.append({
            "side": side,
            "n_complete": int(len(subset)),
            "median_A": float(
                np.nanmedian(aa)
            ),
            "median_C": float(
                np.nanmedian(cc)
            ),
            "median_B": float(
                np.nanmedian(bb)
            ),
            "median_A_C_gap": float(
                np.nanmedian(
                    np.abs(cc - aa)
                )
            ),
            "median_C_B_gap": float(
                np.nanmedian(
                    np.abs(bb - cc)
                )
            ),
            "median_A_B_gap": float(
                np.nanmedian(
                    np.abs(bb - aa)
                )
            ),
            "A_before_C": float(
                np.mean(aa < cc)
            ),
            "C_before_B": float(
                np.mean(cc < bb)
            ),
            "A_before_B": float(
                np.mean(aa < bb)
            ),
            "A_C_B": float(
                np.mean(
                    (aa < cc)
                    &
                    (cc < bb)
                )
            ),
            "B_C_A": float(
                np.mean(
                    (bb < cc)
                    &
                    (cc < aa)
                )
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_orderings(
    expression_df: pd.DataFrame,
    repetitions: int = BOOTSTRAP_REPETITIONS
) -> pd.DataFrame:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    output = []

    n = len(
        expression_df
    )

    if n == 0:
        return pd.DataFrame()

    for side in ("left", "right"):

        column = (
            f"ordering_{side}"
        )

        valid = expression_df[
            expression_df[column]
            .notna()
        ][column].to_numpy()

        if len(valid) == 0:
            continue

        unique_orderings = sorted(
            set(valid)
        )

        bootstrap_values = {
            ordering: []
            for ordering in unique_orderings
        }

        for _ in range(repetitions):

            sample = rng.choice(
                valid,
                size=len(valid),
                replace=True
            )

            counts = pd.Series(
                sample
            ).value_counts(
                normalize=True
            )

            for ordering in unique_orderings:

                bootstrap_values[
                    ordering
                ].append(
                    float(
                        counts.get(
                            ordering,
                            0.0
                        )
                    )
                )

        for ordering in unique_orderings:

            values = np.asarray(
                bootstrap_values[
                    ordering
                ],
                dtype=float
            )

            output.append({
                "side": side,
                "ordering": ordering,
                "median": float(
                    np.median(values)
                ),
                "ci_2_5": float(
                    np.percentile(
                        values,
                        2.5
                    )
                ),
                "ci_97_5": float(
                    np.percentile(
                        values,
                        97.5
                    )
                ),
                "n_bootstrap": repetitions,
            })

    return pd.DataFrame(output)


# ============================================================
# PLOT ONE EXPRESSION
# ============================================================

def plot_expression(
    sequence: pd.DataFrame,
    result: Dict,
    output_path: Path
) -> None:

    viewpoints = (
        sequence["viewpoint"]
        .to_numpy(
            dtype=float
        )
    )

    A = (
        sequence[
            "A_angular_distance_deg"
        ]
        .to_numpy(
            dtype=float
        )
    )

    C = (
        sequence[
            "C_margin"
        ]
        .to_numpy(
            dtype=float
        )
    )

    true_folder = normalize_expression(
        sequence["folder"].iloc[0]
    )

    prediction = (
        sequence[
            "B_predicted_folder"
        ]
        .map(normalize_expression)
        .to_numpy()
    )

    B_correct = (
        prediction == true_folder
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True
    )

    # --------------------------------------------------------
    # A
    # --------------------------------------------------------

    ax = axes[0]

    ax.plot(
        viewpoints,
        A,
        linewidth=1.5
    )

    ax.axhline(
        DEFAULT_THRESHOLDS[
            "A_angular"
        ],
        linestyle="--",
        linewidth=1
    )

    for side in ("left", "right"):

        key = f"A_{side}"

        value = result.get(key)

        if np.isfinite(
            safe_float(value)
        ):
            ax.axvline(
                value,
                linestyle=":"
            )

    ax.set_ylabel(
        "A angular distance"
    )

    ax.set_title(
        "A — Representation drift"
    )

    ax.grid(
        alpha=0.2
    )

    # --------------------------------------------------------
    # C
    # --------------------------------------------------------

    ax = axes[1]

    ax.plot(
        viewpoints,
        C,
        linewidth=1.5
    )

    ax.axhline(
        DEFAULT_THRESHOLDS[
            "C_margin"
        ],
        linestyle="--",
        linewidth=1
    )

    for side in ("left", "right"):

        value = result.get(
            f"C_{side}"
        )

        if np.isfinite(
            safe_float(value)
        ):
            ax.axvline(
                value,
                linestyle=":"
            )

    ax.set_ylabel(
        "C margin"
    )

    ax.set_title(
        "C — Expression separability"
    )

    ax.grid(
        alpha=0.2
    )

    # --------------------------------------------------------
    # B
    # --------------------------------------------------------

    ax = axes[2]

    ax.plot(
        viewpoints,
        B_correct.astype(float),
        drawstyle="steps-mid",
        linewidth=1.5
    )

    ax.set_ylim(
        -0.1,
        1.1
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels(
        [
            "failure",
            "correct"
        ]
    )

    for side in ("left", "right"):

        value = result.get(
            f"B_{side}"
        )

        if np.isfinite(
            safe_float(value)
        ):
            ax.axvline(
                value,
                linestyle=":"
            )

    ax.set_xlabel(
        "Viewpoint"
    )

    ax.set_ylabel(
        "B"
    )

    ax.set_title(
        "B — Expression consistency"
    )

    ax.grid(
        alpha=0.2
    )

    expression = normalize_expression(
        sequence["expression"].iloc[0]
    )

    fig.suptitle(
        f"{expression} | {true_folder}",
        fontsize=14
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# PLOT POPULATION
# ============================================================

def plot_population(
    expression_df: pd.DataFrame
) -> None:

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Boundary distributions
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    data = []

    labels = []

    for side in ("left", "right"):

        for metric in ("A", "C", "B"):

            values = pd.to_numeric(
                expression_df[
                    f"{metric}_{side}"
                ],
                errors="coerce"
            ).dropna()

            if len(values) > 0:

                data.append(
                    values.to_numpy()
                )

                labels.append(
                    f"{metric}-{side}"
                )

    if data:

        ax.boxplot(
            data,
            tick_labels=labels
        )

    ax.set_ylabel(
        "Boundary viewpoint"
    )

    ax.set_title(
        "A / C / B boundary distributions"
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR /
        "boundary_distributions.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # A-C-B scatter
    # --------------------------------------------------------

    for side in ("left", "right"):

        complete = expression_df[
            [
                f"A_{side}",
                f"C_{side}",
                f"B_{side}",
            ]
        ].dropna()

        if complete.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(8, 7)
        )

        ax.scatter(
            complete[
                f"A_{side}"
            ],
            complete[
                f"C_{side}"
            ],
            alpha=0.5,
            label="A vs C"
        )

        ax.set_xlabel(
            "A boundary"
        )

        ax.set_ylabel(
            "C boundary"
        )

        ax.set_title(
            f"A vs C boundaries — {side}"
        )

        ax.grid(
            alpha=0.2
        )

        fig.tight_layout()

        fig.savefig(
            PLOT_DIR /
            f"A_vs_C_{side}.png",
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        # ----------------------------------------------------
        # C vs B
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(8, 7)
        )

        ax.scatter(
            complete[
                f"C_{side}"
            ],
            complete[
                f"B_{side}"
            ],
            alpha=0.5
        )

        ax.set_xlabel(
            "C boundary"
        )

        ax.set_ylabel(
            "B boundary"
        )

        ax.set_title(
            f"C vs B boundaries — {side}"
        )

        ax.grid(
            alpha=0.2
        )

        fig.tight_layout()

        fig.savefig(
            PLOT_DIR /
            f"C_vs_B_{side}.png",
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)


# ============================================================
# README
# ============================================================

def write_readme(
    thresholds: Dict[str, float],
    diagnostics: pd.DataFrame,
    directional: pd.DataFrame
) -> None:

    lines = []

    lines.append(
        "# A / C / B Ordering Diagnostic Analysis"
    )

    lines.append("")

    lines.append(
        "This analysis does NOT assume A < C < B."
    )

    lines.append("")

    lines.append(
        "A = representation drift"
    )

    lines.append(
        "C = expression separability"
    )

    lines.append(
        "B = expression prediction consistency"
    )

    lines.append("")

    lines.append(
        "The analysis follows each complete 215-viewpoint "
        "expression sequence."
    )

    lines.append("")

    lines.append(
        "A boundary is detected when the angular representation "
        "distance exceeds the robust A threshold for "
        f"{SUSTAINED_COUNT} consecutive viewpoints."
    )

    lines.append("")

    lines.append(
        "C boundary is detected when C_margin falls below "
        f"the robust C threshold for {SUSTAINED_COUNT} "
        "consecutive viewpoints."
    )

    lines.append("")

    lines.append(
        "B boundary is detected when prediction differs from "
        "the true expression folder for "
        f"{SUSTAINED_COUNT} consecutive viewpoints."
    )

    lines.append("")

    lines.append(
        "## Thresholds"
    )

    lines.append("")

    for key, value in thresholds.items():

        lines.append(
            f"- {key}: {value}"
        )

    lines.append("")

    lines.append(
        "## Detection diagnostics"
    )

    lines.append("")

    lines.append(
        diagnostics.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "## Directional summary"
    )

    lines.append("")

    if directional.empty:

        lines.append(
            "No complete A/C/B directional cases."
        )

    else:

        lines.append(
            directional.to_string(
                index=False
            )
        )

    lines.append("")

    lines.append(
        "## Interpretation"
    )

    lines.append("")

    lines.append(
        "The ordering is empirical. The script reports "
        "all observed orderings and does not impose a "
        "preferred causal sequence."
    )

    lines.append("")

    lines.append(
        "A missing boundary means that the corresponding "
        "event was not detected under the current operational "
        "definition; it does not mean that the representation "
        "or model necessarily remained perfectly stable."
    )

    OUTPUT_README.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# REPORT
# ============================================================

def write_report(
    thresholds: Dict[str, float],
    expression_df: pd.DataFrame,
    diagnostics: pd.DataFrame,
    population: pd.DataFrame,
    directional: pd.DataFrame,
    bootstrap: pd.DataFrame
) -> None:

    report = {
        "project_root": str(ROOT),
        "metrics_file": str(METRICS_FILE),

        "configuration": {
            "expected_viewpoints": EXPECTED_VIEWPOINTS,
            "frontal_viewpoint": FRONTAL_VIEWPOINT,
            "baseline_half_width": BASELINE_HALF_WIDTH,
            "sustained_count": SUSTAINED_COUNT,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "primary_A_metric": PRIMARY_A_METRIC,
        },

        "thresholds": thresholds,

        "complete_expressions": int(
            len(expression_df)
        ),

        "diagnostics": (
            diagnostics
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),

        "population_orderings": (
            population
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),

        "directional_summary": (
            directional
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),

        "bootstrap": (
            bootstrap
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),

        "important_note": (
            "A < C < B is not assumed. "
            "All ordering results are empirical."
        ),
    }

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning
    )

    print_header(
        "A / C / B ORDERING DIAGNOSTIC ANALYSIS"
    )

    print(
        f"\nProject root:\n{ROOT}"
    )

    print(
        f"\nAnalysis directory:\n{ANALYSIS_DIR}"
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    thresholds = load_thresholds()

    print_header(
        "THRESHOLDS"
    )

    print(
        f"A angular threshold : "
        f"{thresholds['A_angular']:.8f}"
    )

    print(
        f"C margin threshold  : "
        f"{thresholds['C_margin']:.8f}"
    )

    print(
        f"Sustained viewpoints: "
        f"{SUSTAINED_COUNT}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_metrics()

    # --------------------------------------------------------
    # Complete sequences
    # --------------------------------------------------------

    print_header(
        "SEQUENCE COVERAGE"
    )

    sequences = find_complete_sequences(
        df
    )

    print(
        f"Complete expression sequences: "
        f"{len(sequences)}"
    )

    excluded = (
        df.groupby("folder")["viewpoint"]
        .nunique()
    )

    incomplete = excluded[
        excluded != EXPECTED_VIEWPOINTS
    ]

    print(
        f"Incomplete sequences excluded: "
        f"{len(incomplete)}"
    )

    if len(incomplete) > 0:

        for folder, count in incomplete.items():

            print(
                f"  {folder}: "
                f"{int(count)} viewpoints"
            )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    print_header(
        "ANALYZING A / C / B"
    )

    results = []

    for i, (
        folder,
        sequence
    ) in enumerate(
        sequences.items(),
        start=1
    ):

        result = analyze_expression(
            sequence,
            thresholds
        )

        results.append(
            result
        )

        if (
            i <= 5
            or i % 50 == 0
            or i == len(sequences)
        ):

            print(
                f"Processed "
                f"{i:,}/"
                f"{len(sequences):,}"
            )

    expression_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    print_header(
        "POPULATION ORDERINGS"
    )

    population = population_summary(
        expression_df
    )

    print(
        population.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    print_header(
        "DETECTION DIAGNOSTICS"
    )

    diagnostics = detection_diagnostics(
        expression_df
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Directional
    # --------------------------------------------------------

    print_header(
        "DIRECTIONAL ANALYSIS"
    )

    directional = directional_summary(
        expression_df
    )

    if directional.empty:

        print(
            "No complete A/C/B cases."
        )

    else:

        print(
            directional.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    print_header(
        f"BOOTSTRAP ({BOOTSTRAP_REPETITIONS} repetitions)"
    )

    bootstrap = bootstrap_orderings(
        expression_df,
        BOOTSTRAP_REPETITIONS
    )

    if bootstrap.empty:

        print(
            "No bootstrap results."
        )

    else:

        print(
            bootstrap.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save expression table
    # --------------------------------------------------------

    expression_df.to_csv(
        OUTPUT_EXPRESSION,
        index=False
    )

    population.to_csv(
        OUTPUT_POPULATION,
        index=False
    )

    directional.to_csv(
        OUTPUT_DIRECTIONAL,
        index=False
    )

    bootstrap.to_csv(
        OUTPUT_BOOTSTRAP,
        index=False
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print_header(
        "BUILDING POPULATION PLOTS"
    )

    plot_population(
        expression_df
    )

    # --------------------------------------------------------
    # Diagnostic examples
    # --------------------------------------------------------

    print_header(
        "BUILDING EXAMPLE TRAJECTORY PLOTS"
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Pick a few useful examples:
    # 1. expression with complete A/C/B
    # 2. expression with A < C < B if any
    # 3. expression with a different ordering
    # 4. first expression
    #
    # This is deliberately data-driven.

    selected_folders = []

    # First complete A/C/B
    for _, row in expression_df.iterrows():

        if (
            row.get(
                "A_left_detected",
                False
            )
            and row.get(
                "C_left_detected",
                False
            )
            and row.get(
                "B_left_detected",
                False
            )
        ):

            selected_folders.append(
                row["folder"]
            )

            break

    # Try to find A<C<B
    candidate = expression_df[
        expression_df[
            "ordering_left"
        ] == "A<C<B"
    ]

    if not candidate.empty:

        selected_folders.append(
            candidate.iloc[0]["folder"]
        )

    # Try to find something other than A<C<B
    candidate = expression_df[
        ~expression_df[
            "ordering_left"
        ].isin(
            [
                "A<C<B",
                "missing_A",
                "missing_C",
                "missing_B",
                "missing_A_C",
                "missing_A_B",
                "missing_C_B",
                "missing_A_C_B",
            ]
        )
    ]

    if not candidate.empty:

        selected_folders.append(
            candidate.iloc[0]["folder"]
        )

    # Unique
    selected_folders = list(
        dict.fromkeys(
            selected_folders
        )
    )[:5]

    for folder in selected_folders:

        sequence = sequences.get(
            folder
        )

        if sequence is None:
            continue

        row = expression_df[
            expression_df["folder"]
            == folder
        ]

        if row.empty:
            continue

        result = row.iloc[0].to_dict()

        safe_name = (
            str(folder)
            .replace("\\", "_")
            .replace("/", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
        )

        plot_expression(
            sequence,
            result,
            PLOT_DIR /
            f"{safe_name}.png"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    write_report(
        thresholds,
        expression_df,
        diagnostics,
        population,
        directional,
        bootstrap
    )

    write_readme(
        thresholds,
        diagnostics,
        directional
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_header(
        "DONE"
    )

    print(
        f"Complete sequences: "
        f"{len(expression_df):,}"
    )

    print()

    print(
        "Output files:"
    )

    print(
        OUTPUT_EXPRESSION
    )

    print(
        OUTPUT_POPULATION
    )

    print(
        OUTPUT_DIRECTIONAL
    )

    print(
        OUTPUT_BOOTSTRAP
    )

    print(
        OUTPUT_REPORT
    )

    print(
        OUTPUT_README
    )

    print(
        PLOT_DIR
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The script does NOT assume A < C < B."
    )

    print(
        "It reports the ordering actually observed "
        "under the current operational definitions."
    )


if __name__ == "__main__":
    main()