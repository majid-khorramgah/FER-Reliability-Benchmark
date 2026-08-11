# ================================================================
# analyze_statistical_validation.py
#
# FER Reliability Benchmark
#
# Statistical validation of:
#
#   A = Representation Drift
#   C = Expression Separability Collapse
#   B = Expression Consistency / Prediction Failure
#
# This script operates on:
#
#   analysis/per_view_metrics_multimetric.csv
#
# It does NOT load embeddings.
#
# Main goals:
#
#   1. Validate A/C/B boundary differences statistically.
#   2. Compare left-vs-right viewpoint directions.
#   3. Bootstrap confidence intervals.
#   4. Paired permutation tests.
#   5. Effect sizes.
#   6. Early-warning analysis.
#   7. Viewpoint-shuffle null model.
#   8. Multiple-comparison correction.
#   9. Publication-ready plots.
#
# IMPORTANT:
#
# The script does NOT assume:
#
#       A < C < B
#
# It measures what is actually observed.
#
# ================================================================

from __future__ import annotations

import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ================================================================
# CONFIGURATION
# ================================================================

RANDOM_SEED = 20260808

N_BOOTSTRAP = 5000

N_PERMUTATIONS = 10000

N_VIEWPOINT_SHUFFLES = 2000

ALPHA = 0.05

BASELINE_VIEWPOINT = 107

SUSTAINED_VIEWPOINTS = 3

# Current thresholds obtained from the previous analysis.
#
# These are configuration values, NOT hypotheses.
#
# They can be overridden automatically from the existing analysis
# report if available.

DEFAULT_A_ANGULAR_THRESHOLD = 13.43702602

DEFAULT_C_MARGIN_THRESHOLD = 0.00237080


# ================================================================
# PROJECT PATHS
# ================================================================

def project_root() -> Path:
    """
    Project root is:

        D:\1405\FER-Reliability-Benchmark
    """

    return Path(__file__).resolve().parent


def analysis_dir() -> Path:
    return project_root() / "analysis"


def input_csv() -> Path:
    return analysis_dir() / "4_analyze_embeddings_trajectory" / "per_view_metrics_multimetric.csv"


def output_dir() -> Path:
    path = analysis_dir() / "7_analyze_statistical_validation"

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ================================================================
# RANDOMNESS
# ================================================================

def initialize_randomness() -> None:

    random.seed(RANDOM_SEED)

    np.random.seed(RANDOM_SEED)


# ================================================================
# COLUMN HELPERS
# ================================================================

def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:

    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in normalized:

            return normalized[key]

    # More flexible matching
    for column in df.columns:

        low = str(column).lower()

        for candidate in candidates:

            if candidate.lower() in low:

                return column

    if required:

        raise RuntimeError(
            "\nRequired column not found.\n"
            f"Candidates: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


# ================================================================
# LOAD DATA
# ================================================================

def load_data() -> pd.DataFrame:

    path = input_csv()

    if not path.exists():

        raise FileNotFoundError(
            f"\nInput file does not exist:\n{path}"
        )

    print("=" * 70)
    print("STATISTICAL VALIDATION")
    print("=" * 70)

    print()
    print("Project root:")
    print(project_root())

    print()
    print("Analysis directory:")
    print(analysis_dir())

    print()
    print("Loading:")
    print(path)

    df = pd.read_csv(path)

    print()
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns found: {len(df.columns)}")

    return df


# ================================================================
# STANDARDIZE COLUMNS
# ================================================================

def standardize_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:

    mapping = {}

    mapping["expression"] = find_column(
        df,
        [
            "expression",
        ],
    )

    mapping["folder"] = find_column(
        df,
        [
            "folder",
        ],
    )

    mapping["viewpoint"] = find_column(
        df,
        [
            "viewpoint",
        ],
    )

    mapping["A"] = find_column(
        df,
        [
            "A_angular_distance_deg",
            "A_angular",
            "angular_distance_deg",
        ],
    )

    mapping["C"] = find_column(
        df,
        [
            "C_margin",
        ],
    )

    mapping["B"] = find_column(
        df,
        [
            "B_predicted_folder",
            "predicted_folder",
        ],
    )

    print()
    print("=" * 70)
    print("COLUMN MAPPING")
    print("=" * 70)

    for key, value in mapping.items():

        print(
            f"{key:<20}: {value}"
        )

    out = pd.DataFrame()

    for key, column in mapping.items():

        out[key] = df[column]

    # Keep original metadata if useful.
    out["source_row"] = np.arange(len(out))

    # Numeric conversion
    out["viewpoint"] = pd.to_numeric(
        out["viewpoint"],
        errors="coerce",
    )

    out["A"] = pd.to_numeric(
        out["A"],
        errors="coerce",
    )

    out["C"] = pd.to_numeric(
        out["C"],
        errors="coerce",
    )

    out["expression"] = (
        out["expression"]
        .astype(str)
        .str.strip()
    )

    out["folder"] = (
        out["folder"]
        .astype(str)
        .str.strip()
    )

    out["B"] = (
        out["B"]
        .astype(str)
        .str.strip()
    )

    return out, mapping


# ================================================================
# LOAD THRESHOLDS
# ================================================================

def load_previous_thresholds() -> Dict[str, float]:

    thresholds = {
        "A": DEFAULT_A_ANGULAR_THRESHOLD,
        "C": DEFAULT_C_MARGIN_THRESHOLD,
    }

    report_path = (
        analysis_dir()
        / "analysis_report.json"
    )

    trajectory_report = (
        analysis_dir()
        / "trajectory_analysis_report.json"
    )

    candidate_reports = [
        report_path,
        trajectory_report,
        analysis_dir()
        / "analysis_report_fixed.json",
    ]

    for path in candidate_reports:

        if not path.exists():

            continue

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(f)

            # Recursive search
            def recursive_search(
                obj,
                keys,
            ):

                if isinstance(obj, dict):

                    for key, value in obj.items():

                        low = str(key).lower()

                        if any(
                            token in low
                            for token in keys
                        ):

                            if isinstance(
                                value,
                                (int, float),
                            ):

                                return float(value)

                        result = recursive_search(
                            value,
                            keys,
                        )

                        if result is not None:

                            return result

                elif isinstance(obj, list):

                    for item in obj:

                        result = recursive_search(
                            item,
                            keys,
                        )

                        if result is not None:

                            return result

                return None

            a_value = recursive_search(
                data,
                [
                    "a_angular_threshold",
                    "a angular threshold",
                    "angular threshold",
                ],
            )

            c_value = recursive_search(
                data,
                [
                    "c_margin_threshold",
                    "c margin threshold",
                    "margin threshold",
                ],
            )

            if a_value is not None:

                thresholds["A"] = a_value

            if c_value is not None:

                thresholds["C"] = c_value

        except Exception:

            continue

    return thresholds


# ================================================================
# BUILD COMPLETE EXPRESSION SEQUENCES
# ================================================================

def get_complete_sequences(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:

    sequences = {}

    grouped = df.groupby(
        "folder",
        sort=False,
    )

    for folder, group in grouped:

        viewpoints = set(
            pd.to_numeric(
                group["viewpoint"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

        expected = set(
            range(
                0,
                215,
            )
        )

        if viewpoints == expected:

            sequences[folder] = (
                group
                .sort_values("viewpoint")
                .copy()
            )

    return sequences


# ================================================================
# B FAILURE DEFINITION
# ================================================================

def build_B_failure(
    sequence: pd.DataFrame,
) -> pd.Series:

    """
    B is prediction failure.

    A prediction is considered failed when:

        predicted folder != true folder

    We keep the raw failure signal.

    Boundary detection later requires the failure to remain
    active for SUSTAINED_VIEWPOINTS consecutive viewpoints.
    """

    true_label = (
        sequence["folder"]
        .astype(str)
        .str.strip()
    )

    predicted = (
        sequence["B"]
        .astype(str)
        .str.strip()
    )

    failure = (
        predicted != true_label
    )

    # Strings such as "nan" are not valid predictions.
    invalid_prediction = (
        predicted.str.lower().isin(
            [
                "nan",
                "none",
                "",
            ]
        )
    )

    failure = (
        failure
        & ~invalid_prediction
    )

    return failure


# ================================================================
# SUSTAINED TRUE DETECTION
# ================================================================

def first_sustained_detection(
    viewpoints: np.ndarray,
    signal: np.ndarray,
    direction: str,
    sustained: int = SUSTAINED_VIEWPOINTS,
) -> Optional[int]:

    """
    Find first viewpoint where a signal remains active for
    'sustained' consecutive viewpoints.

    Left:
        106,105,104,...

    Right:
        108,109,110,...

    The baseline 107 is excluded.
    """

    if len(viewpoints) == 0:

        return None

    order = np.argsort(viewpoints)

    viewpoints = viewpoints[order]
    signal = signal[order]

    if direction == "right":

        valid = viewpoints > BASELINE_VIEWPOINT

        viewpoints = viewpoints[valid]
        signal = signal[valid]

        expected_step = 1

    else:

        valid = viewpoints < BASELINE_VIEWPOINT

        viewpoints = viewpoints[valid]
        signal = signal[valid]

        expected_step = -1

        viewpoints = viewpoints[::-1]
        signal = signal[::-1]

    if len(viewpoints) < sustained:

        return None

    for i in range(
        len(viewpoints) - sustained + 1
    ):

        vp = viewpoints[
            i : i + sustained
        ]

        active = signal[
            i : i + sustained
        ]

        if not np.all(active):

            continue

        differences = np.diff(vp)

        if direction == "right":

            contiguous = np.all(
                differences == 1
            )

        else:

            # Left side is traversed from the frontal view outward:
            # 106, 105, 104, ...
            # Therefore consecutive viewpoints differ by -1.
            contiguous = np.all(
                differences == -1
            )

        if contiguous:

            return int(vp[0])

    return None


# ================================================================
# DETECT A BOUNDARY
# ================================================================

def detect_A_boundary(
    sequence: pd.DataFrame,
    threshold: float,
    direction: str,
) -> Optional[int]:

    signal = (
        sequence["A"].to_numpy(
            dtype=float
        )
        >= threshold
    )

    viewpoints = (
        sequence["viewpoint"]
        .to_numpy(dtype=int)
    )

    return first_sustained_detection(
        viewpoints,
        signal,
        direction,
    )


# ================================================================
# DETECT C BOUNDARY
# ================================================================

def detect_C_boundary(
    sequence: pd.DataFrame,
    threshold: float,
    direction: str,
) -> Optional[int]:

    """
    C_margin is assumed to represent the margin between the
    target expression and the closest competing expression.

    Therefore smaller margin = worse separability.

    Collapse is:

        C_margin <= threshold
    """

    signal = (
        sequence["C"].to_numpy(
            dtype=float
        )
        <= threshold
    )

    viewpoints = (
        sequence["viewpoint"]
        .to_numpy(dtype=int)
    )

    return first_sustained_detection(
        viewpoints,
        signal,
        direction,
    )


# ================================================================
# DETECT B BOUNDARY
# ================================================================

def detect_B_boundary(
    sequence: pd.DataFrame,
    direction: str,
) -> Optional[int]:

    failure = build_B_failure(
        sequence
    )

    viewpoints = (
        sequence["viewpoint"]
        .to_numpy(dtype=int)
    )

    return first_sustained_detection(
        viewpoints,
        failure.to_numpy(
            dtype=bool
        ),
        direction,
    )


# ================================================================
# DISTANCE FROM FRONTAL VIEW
# ================================================================

def boundary_distance(
    boundary: Optional[int],
    direction: str,
) -> float:

    if boundary is None:

        return np.nan

    return abs(
        int(boundary)
        - BASELINE_VIEWPOINT
    )


# ================================================================
# ORDERING
# ================================================================

def ordering_label(
    A: float,
    C: float,
    B: float,
) -> str:

    if not (
        np.isfinite(A)
        and np.isfinite(C)
        and np.isfinite(B)
    ):

        return "incomplete"

    values = {
        "A": A,
        "C": C,
        "B": B,
    }

    # Tolerance for practically identical boundaries.
    tolerance = 0.5

    sorted_items = sorted(
        values.items(),
        key=lambda x: x[1],
    )

    groups = []

    current = [
        sorted_items[0][0]
    ]

    current_value = (
        sorted_items[0][1]
    )

    for name, value in sorted_items[1:]:

        if abs(
            value - current_value
        ) <= tolerance:

            current.append(name)

        else:

            groups.append(
                "=".join(current)
            )

            current = [name]

            current_value = value

    groups.append(
        "=".join(current)
    )

    return "<".join(groups)


# ================================================================
# BUILD BOUNDARY TABLE
# ================================================================

def build_boundary_table(
    sequences: Dict[str, pd.DataFrame],
    thresholds: Dict[str, float],
) -> pd.DataFrame:

    rows = []

    print()
    print("=" * 70)
    print("BUILDING A / C / B BOUNDARIES")
    print("=" * 70)

    total = len(sequences)

    for index, (
        folder,
        sequence,
    ) in enumerate(
        sequences.items(),
        start=1,
    ):

        if (
            index <= 5
            or index % 50 == 0
            or index == total
        ):

            print(
                f"Processed {index}/{total}"
            )

        expression = str(
            sequence["expression"].iloc[0]
        )

        for direction in [
            "left",
            "right",
        ]:

            A_boundary = detect_A_boundary(
                sequence,
                thresholds["A"],
                direction,
            )

            C_boundary = detect_C_boundary(
                sequence,
                thresholds["C"],
                direction,
            )

            B_boundary = detect_B_boundary(
                sequence,
                direction,
            )

            A_distance = boundary_distance(
                A_boundary,
                direction,
            )

            C_distance = boundary_distance(
                C_boundary,
                direction,
            )

            B_distance = boundary_distance(
                B_boundary,
                direction,
            )

            rows.append(
                {
                    "folder": folder,
                    "expression": expression,
                    "direction": direction,

                    "A_boundary": A_boundary,
                    "C_boundary": C_boundary,
                    "B_boundary": B_boundary,

                    "A_distance": A_distance,
                    "C_distance": C_distance,
                    "B_distance": B_distance,

                    "A_C_gap": (
                        C_distance - A_distance
                        if np.isfinite(A_distance)
                        and np.isfinite(C_distance)
                        else np.nan
                    ),

                    "C_B_gap": (
                        B_distance - C_distance
                        if np.isfinite(C_distance)
                        and np.isfinite(B_distance)
                        else np.nan
                    ),

                    "A_B_gap": (
                        B_distance - A_distance
                        if np.isfinite(A_distance)
                        and np.isfinite(B_distance)
                        else np.nan
                    ),

                    "ordering": ordering_label(
                        A_distance,
                        C_distance,
                        B_distance,
                    ),
                }
            )

    return pd.DataFrame(rows)


# ================================================================
# BASIC STATISTICS
# ================================================================

def median_and_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOTSTRAP,
) -> Dict[str, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return {
            "n": 0,
            "median": np.nan,
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    median = float(
        np.median(values)
    )

    mean = float(
        np.mean(values)
    )

    boot = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot[i] = np.median(
            sample
        )

    return {
        "n": int(len(values)),
        "median": median,
        "mean": mean,
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


# ================================================================
# EFFECT SIZE
# ================================================================

def paired_rank_biserial(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    """
    Paired rank-biserial correlation.

    Positive means x > y tends to dominate.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    d = x[valid] - y[valid]

    d = d[
        d != 0
    ]

    if len(d) == 0:

        return 0.0

    positive = np.sum(
        d > 0
    )

    negative = np.sum(
        d < 0
    )

    return float(
        (
            positive - negative
        )
        / len(d)
    )


# ================================================================
# PAIRED PERMUTATION TEST
# ================================================================

def paired_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = N_PERMUTATIONS,
) -> Dict[str, float]:

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) == 0:

        return {
            "n": 0,
            "observed_median_difference": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
        }

    differences = x - y

    observed = float(
        np.median(differences)
    )

    effect = paired_rank_biserial(
        x,
        y,
    )

    count = 0

    for _ in range(n_perm):

        signs = rng.choice(
            np.array(
                [-1.0, 1.0]
            ),
            size=len(differences),
        )

        permuted = (
            differences
            * signs
        )

        statistic = np.median(
            permuted
        )

        if abs(statistic) >= abs(
            observed
        ):

            count += 1

    p = (
        count + 1
    ) / (
        n_perm + 1
    )

    return {
        "n": int(len(x)),
        "observed_median_difference": observed,
        "p_value": float(p),
        "effect_size_rank_biserial": effect,
    }


# ================================================================
# PAIRED BOOTSTRAP DIFFERENCE
# ================================================================

def paired_bootstrap_difference(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOTSTRAP,
) -> Dict[str, float]:

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) == 0:

        return {
            "n": 0,
            "median_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    difference = (
        x - y
    )

    observed = float(
        np.median(
            difference
        )
    )

    boot = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):

        indices = rng.integers(
            0,
            len(x),
            size=len(x),
        )

        boot[i] = np.median(
            difference[
                indices
            ]
        )

    return {
        "n": int(len(x)),
        "median_difference": observed,
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


# ================================================================
# LEFT / RIGHT STATISTICS
# ================================================================

def analyze_directional_boundaries(
    boundary_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:

    rows = []

    for direction in [
        "left",
        "right",
    ]:

        subset = boundary_df[
            boundary_df["direction"]
            == direction
        ]

        print()
        print("=" * 70)
        print(
            f"{direction.upper()} BOUNDARY STATISTICS"
        )
        print("=" * 70)

        for metric in [
            "A_distance",
            "C_distance",
            "B_distance",
            "A_C_gap",
            "C_B_gap",
            "A_B_gap",
        ]:

            stats = median_and_ci(
                subset[metric]
                .to_numpy(),
                rng,
            )

            row = {
                "direction": direction,
                "metric": metric,
                **stats,
            }

            rows.append(row)

            print(
                f"{metric:<15} "
                f"n={stats['n']:<4} "
                f"median={stats['median']:.3f} "
                f"95% CI="
                f"[{stats['ci_low']:.3f}, "
                f"{stats['ci_high']:.3f}]"
            )

    return pd.DataFrame(rows)


# ================================================================
# LEFT VS RIGHT PAIRED TEST
# ================================================================

def compare_left_right(
    boundary_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("PAIRED LEFT vs RIGHT COMPARISON")
    print("=" * 70)

    pivot = boundary_df.pivot(
        index="folder",
        columns="direction",
    )

    rows = []

    for metric in [
        "A_distance",
        "C_distance",
        "B_distance",
        "A_C_gap",
        "C_B_gap",
        "A_B_gap",
    ]:

        if (
            (metric, "left")
            not in pivot.columns
            or
            (metric, "right")
            not in pivot.columns
        ):

            continue

        left = pivot[
            (metric, "left")
        ].to_numpy()

        right = pivot[
            (metric, "right")
        ].to_numpy()

        perm = paired_permutation_test(
            left,
            right,
            rng,
        )

        boot = paired_bootstrap_difference(
            left,
            right,
            rng,
        )

        row = {
            "metric": metric,

            "n": perm["n"],

            "left_minus_right_median":
                perm[
                    "observed_median_difference"
                ],

            "p_value":
                perm["p_value"],

            "effect_size":
                perm[
                    "effect_size_rank_biserial"
                ],

            "bootstrap_ci_low":
                boot["ci_low"],

            "bootstrap_ci_high":
                boot["ci_high"],
        }

        rows.append(row)

        print(
            f"{metric:<15} "
            f"median(L-R)="
            f"{row['left_minus_right_median']:.3f} "
            f"p={row['p_value']:.6f} "
            f"effect="
            f"{row['effect_size']:.3f}"
        )

    return pd.DataFrame(rows)


# ================================================================
# EARLY WARNING
# ================================================================

def early_warning_analysis(
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    print()
    print("=" * 70)
    print("EARLY WARNING ANALYSIS")
    print("=" * 70)

    for direction in [
        "left",
        "right",
    ]:

        subset = boundary_df[
            boundary_df["direction"]
            == direction
        ]

        print()
        print(
            direction.upper()
        )

        tests = {
            "A_before_B":
                subset["A_distance"]
                < subset["B_distance"],

            "C_before_B":
                subset["C_distance"]
                < subset["B_distance"],

            "A_before_C":
                subset["A_distance"]
                < subset["C_distance"],

            "A_before_B_and_C":
                (
                    subset["A_distance"]
                    < subset["B_distance"]
                )
                &
                (
                    subset["A_distance"]
                    < subset["C_distance"]
                ),
        }

        for name, condition in tests.items():

            valid = (
                np.isfinite(
                    subset["A_distance"]
                )
                &
                np.isfinite(
                    subset["C_distance"]
                )
                &
                np.isfinite(
                    subset["B_distance"]
                )
            )

            values = (
                condition[
                    valid
                ]
                .to_numpy(
                    dtype=bool
                )
            )

            n = len(values)

            if n == 0:

                proportion = np.nan

            else:

                proportion = (
                    np.mean(values)
                )

            # Wilson-like bootstrap CI
            if n > 0:

                rng = np.random.default_rng(
                    RANDOM_SEED
                    + len(rows)
                    + 17
                )

                boot = []

                for _ in range(
                    N_BOOTSTRAP
                ):

                    sample = rng.choice(
                        values,
                        size=n,
                        replace=True,
                    )

                    boot.append(
                        np.mean(sample)
                    )

                ci_low = np.percentile(
                    boot,
                    2.5,
                )

                ci_high = np.percentile(
                    boot,
                    97.5,
                )

            else:

                ci_low = np.nan
                ci_high = np.nan

            rows.append(
                {
                    "direction":
                        direction,

                    "test":
                        name,

                    "n":
                        n,

                    "proportion":
                        proportion,

                    "ci_low":
                        ci_low,

                    "ci_high":
                        ci_high,
                }
            )

            print(
                f"{name:<25} "
                f"{proportion:.3%} "
                f"95% CI="
                f"[{ci_low:.3%}, "
                f"{ci_high:.3%}]"
            )

    return pd.DataFrame(rows)


# ================================================================
# ORDERING ANALYSIS
# ================================================================

def ordering_analysis(
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("OBSERVED A / C / B ORDERINGS")
    print("=" * 70)

    rows = []

    for direction in [
        "left",
        "right",
    ]:

        subset = boundary_df[
            boundary_df["direction"]
            == direction
        ]

        counts = (
            subset["ordering"]
            .value_counts(
                dropna=False
            )
        )

        denominator = len(subset)

        print()
        print(
            direction.upper()
        )

        for ordering, count in counts.items():

            proportion = (
                count
                / denominator
            )

            print(
                f"{str(ordering):<15} "
                f"{count:>4} "
                f"{proportion:.3%}"
            )

            rows.append(
                {
                    "direction":
                        direction,

                    "ordering":
                        ordering,

                    "count":
                        int(count),

                    "proportion":
                        float(proportion),
                }
            )

    return pd.DataFrame(rows)


# ================================================================
# VIEWPOINT SHUFFLE NULL MODEL
# ================================================================

def shuffled_boundary(
    sequence: pd.DataFrame,
    rng: np.random.Generator,
    direction: str,
    threshold_A: float,
    threshold_C: float,
) -> Tuple[
    Optional[int],
    Optional[int],
    Optional[int],
]:

    """
    Null model:

    Keep all metric values from an expression but randomly
    permute their association with viewpoint.

    This destroys the true viewpoint trajectory while preserving
    the marginal distribution of A/C/B.

    """

    sequence = sequence.copy()

    n = len(sequence)

    permutation = rng.permutation(n)

    shuffled = sequence.copy()

    shuffled["A"] = (
        sequence["A"]
        .to_numpy()[permutation]
    )

    shuffled["C"] = (
        sequence["C"]
        .to_numpy()[permutation]
    )

    shuffled["B"] = (
        sequence["B"]
        .to_numpy()[permutation]
    )

    A = detect_A_boundary(
        shuffled,
        threshold_A,
        direction,
    )

    C = detect_C_boundary(
        shuffled,
        threshold_C,
        direction,
    )

    B = detect_B_boundary(
        shuffled,
        direction,
    )

    return A, C, B


# ================================================================
# SHUFFLE NULL MODEL
# ================================================================

def viewpoint_shuffle_test(
    sequences: Dict[str, pd.DataFrame],
    observed_boundary_df: pd.DataFrame,
    thresholds: Dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("VIEWPOINT-SHUFFLE NULL MODEL")
    print("=" * 70)

    observed_rows = []

    for direction in [
        "left",
        "right",
    ]:

        subset = observed_boundary_df[
            observed_boundary_df["direction"]
            == direction
        ]

        valid = (
            subset[
                [
                    "A_distance",
                    "C_distance",
                    "B_distance",
                ]
            ]
            .notna()
            .all(axis=1)
        )

        subset = subset[
            valid
        ]

        observed_A_before_B = np.mean(
            subset["A_distance"]
            < subset["B_distance"]
        )

        observed_A_C_B = np.mean(
            (
                subset["A_distance"]
                < subset["C_distance"]
            )
            &
            (
                subset["C_distance"]
                < subset["B_distance"]
            )
        )

        observed_rows.append(
            {
                "direction":
                    direction,

                "observed_A_before_B":
                    observed_A_before_B,

                "observed_A_C_B":
                    observed_A_C_B,
            }
        )

    observed_df = pd.DataFrame(
        observed_rows
    )

    null_rows = []

    folders = list(
        sequences.keys()
    )

    for shuffle_id in range(
        N_VIEWPOINT_SHUFFLES
    ):

        if (
            shuffle_id == 0
            or
            shuffle_id % 100 == 0
        ):

            print(
                f"Shuffle "
                f"{shuffle_id + 1}/"
                f"{N_VIEWPOINT_SHUFFLES}"
            )

        sampled_folders = rng.choice(
            folders,
            size=len(folders),
            replace=False,
        )

        for direction in [
            "left",
            "right",
        ]:

            A_before_B_count = 0

            A_C_B_count = 0

            valid_count = 0

            for folder in sampled_folders:

                sequence = sequences[
                    folder
                ]

                A, C, B = shuffled_boundary(
                    sequence,
                    rng,
                    direction,
                    thresholds["A"],
                    thresholds["C"],
                )

                if (
                    A is None
                    or C is None
                    or B is None
                ):

                    continue

                A_distance = boundary_distance(
                    A,
                    direction,
                )

                C_distance = boundary_distance(
                    C,
                    direction,
                )

                B_distance = boundary_distance(
                    B,
                    direction,
                )

                valid_count += 1

                if (
                    A_distance
                    < B_distance
                ):

                    A_before_B_count += 1

                if (
                    A_distance
                    < C_distance
                    < B_distance
                ):

                    A_C_B_count += 1

            if valid_count > 0:

                null_A_B = (
                    A_before_B_count
                    / valid_count
                )

                null_A_C_B = (
                    A_C_B_count
                    / valid_count
                )

            else:

                null_A_B = np.nan
                null_A_C_B = np.nan

            null_rows.append(
                {
                    "shuffle":
                        shuffle_id,

                    "direction":
                        direction,

                    "A_before_B":
                        null_A_B,

                    "A_C_B":
                        null_A_C_B,

                    "n_valid":
                        valid_count,
                }
            )

    null_df = pd.DataFrame(
        null_rows
    )

    result_rows = []

    for direction in [
        "left",
        "right",
    ]:

        observed = observed_df[
            observed_df["direction"]
            == direction
        ].iloc[0]

        null = null_df[
            null_df["direction"]
            == direction
        ]

        for metric in [
            "A_before_B",
            "A_C_B",
        ]:

            values = null[
                metric
            ].dropna().to_numpy()

            observed_value = float(
                observed[
                    (
                        "observed_"
                        + metric
                    )
                ]
            )

            if len(values) > 0:

                p_upper = (
                    np.sum(
                        values
                        >= observed_value
                    ) + 1
                ) / (
                    len(values) + 1
                )

                p_lower = (
                    np.sum(
                        values
                        <= observed_value
                    ) + 1
                ) / (
                    len(values) + 1
                )

                p_two = min(
                    1.0,
                    2
                    * min(
                        p_upper,
                        p_lower,
                    ),
                )

                null_median = np.median(
                    values
                )

                null_low = np.percentile(
                    values,
                    2.5,
                )

                null_high = np.percentile(
                    values,
                    97.5,
                )

            else:

                p_two = np.nan
                null_median = np.nan
                null_low = np.nan
                null_high = np.nan

            result_rows.append(
                {
                    "direction":
                        direction,

                    "metric":
                        metric,

                    "observed":
                        observed_value,

                    "null_median":
                        null_median,

                    "null_ci_low":
                        null_low,

                    "null_ci_high":
                        null_high,

                    "p_value":
                        p_two,

                    "n_shuffles":
                        N_VIEWPOINT_SHUFFLES,
                }
            )

    result = pd.DataFrame(
        result_rows
    )

    return result


# ================================================================
# BINOMIAL-LIKE TEST FOR EARLY WARNING
# ================================================================

def exact_sign_test(
    condition: np.ndarray,
) -> float:

    condition = np.asarray(
        condition,
        dtype=bool,
    )

    n = len(condition)

    if n == 0:

        return np.nan

    k = int(
        np.sum(condition)
    )

    # Exact two-sided sign test
    probabilities = []

    for i in range(
        n + 1
    ):

        probability = (
            math.comb(n, i)
            * (0.5 ** n)
        )

        probabilities.append(
            probability
        )

    observed_probability = probabilities[
        k
    ]

    p = sum(
        probability
        for probability in probabilities
        if probability
        <= observed_probability
        + 1e-15
    )

    return min(
        1.0,
        p,
    )


# ================================================================
# EARLY WARNING SIGNIFICANCE
# ================================================================

def early_warning_significance(
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    print()
    print("=" * 70)
    print("EARLY-WARNING SIGNIFICANCE")
    print("=" * 70)

    for direction in [
        "left",
        "right",
    ]:

        subset = boundary_df[
            boundary_df["direction"]
            == direction
        ].copy()

        valid = (
            subset[
                [
                    "A_distance",
                    "C_distance",
                    "B_distance",
                ]
            ]
            .notna()
            .all(axis=1)
        )

        subset = subset[
            valid
        ]

        tests = {
            "A_before_B":
                (
                    subset["A_distance"]
                    < subset["B_distance"]
                ),

            "C_before_B":
                (
                    subset["C_distance"]
                    < subset["B_distance"]
                ),

            "A_before_C":
                (
                    subset["A_distance"]
                    < subset["C_distance"]
                ),
        }

        for name, condition in tests.items():

            values = condition.to_numpy(
                dtype=bool
            )

            p = exact_sign_test(
                values
            )

            rows.append(
                {
                    "direction":
                        direction,

                    "test":
                        name,

                    "n":
                        len(values),

                    "successes":
                        int(
                            np.sum(values)
                        ),

                    "proportion":
                        float(
                            np.mean(values)
                        ),

                    "sign_test_p":
                        p,
                }
            )

            print(
                f"{direction:<6} "
                f"{name:<15} "
                f"{np.mean(values):.3%} "
                f"p={p:.6f}"
            )

    return pd.DataFrame(
        rows
    )


# ================================================================
# BENJAMINI-HOCHBERG FDR
# ================================================================

def benjamini_hochberg(
    p_values: pd.Series,
) -> np.ndarray:

    p = np.asarray(
        p_values,
        dtype=float,
    )

    result = np.full(
        len(p),
        np.nan,
    )

    valid = np.isfinite(p)

    pv = p[
        valid
    ]

    if len(pv) == 0:

        return result

    order = np.argsort(
        pv
    )

    sorted_p = pv[
        order
    ]

    m = len(
        sorted_p
    )

    adjusted = (
        sorted_p
        * m
        / np.arange(
            1,
            m + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.minimum(
        adjusted,
        1.0,
    )

    temp = np.empty(
        len(pv)
    )

    temp[
        order
    ] = adjusted

    result[
        valid
    ] = temp

    return result


# ================================================================
# APPLY MULTIPLE COMPARISON CORRECTION
# ================================================================

def add_fdr(
    df: pd.DataFrame,
    p_column: str = "p_value",
) -> pd.DataFrame:

    if p_column not in df.columns:

        return df

    df = df.copy()

    df["p_value_fdr"] = (
        benjamini_hochberg(
            df[p_column]
        )
    )

    df["significant_fdr"] = (
        df["p_value_fdr"]
        < ALPHA
    )

    return df


# ================================================================
# SAVE ORDERING SUMMARY
# ================================================================

def ordering_comparison(
    boundary_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for direction in [
        "left",
        "right",
    ]:

        subset = boundary_df[
            boundary_df["direction"]
            == direction
        ]

        complete = subset[
            subset["ordering"]
            != "incomplete"
        ]

        n = len(
            complete
        )

        if n == 0:

            continue

        for ordering in sorted(
            complete["ordering"]
            .unique()
        ):

            count = int(
                np.sum(
                    complete["ordering"]
                    == ordering
                )
            )

            rows.append(
                {
                    "direction":
                        direction,

                    "ordering":
                        ordering,

                    "count":
                        count,

                    "proportion":
                        count / n,
                }
            )

    return pd.DataFrame(
        rows
    )


# ================================================================
# PLOT: BOUNDARY DISTRIBUTIONS
# ================================================================

def plot_boundary_distributions(
    boundary_df: pd.DataFrame,
    out_path: Path,
) -> None:

    metrics = [
        (
            "A_distance",
            "A — Representation Drift",
        ),
        (
            "C_distance",
            "C — Expression Separability",
        ),
        (
            "B_distance",
            "B — Expression Consistency",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    for ax, (
        column,
        title,
    ) in zip(
        axes,
        metrics,
    ):

        data = []

        labels = []

        for direction in [
            "left",
            "right",
        ]:

            values = (
                boundary_df[
                    boundary_df["direction"]
                    == direction
                ][column]
                .dropna()
                .to_numpy()
            )

            data.append(
                values
            )

            labels.append(
                direction
            )

        ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
        )

        ax.set_title(
            title
        )

        ax.set_ylabel(
            "Critical viewpoint distance from V107"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    fig.suptitle(
        "Critical Boundary Distributions"
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ================================================================
# PLOT: LEFT VS RIGHT
# ================================================================

def plot_left_right(
    boundary_df: pd.DataFrame,
    out_path: Path,
) -> None:

    metrics = [
        "A_distance",
        "C_distance",
        "B_distance",
    ]

    labels = [
        "A",
        "C",
        "B",
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = np.arange(
        len(metrics)
    )

    width = 0.35

    left_medians = []

    right_medians = []

    for metric in metrics:

        left = boundary_df[
            boundary_df["direction"]
            == "left"
        ][metric].dropna()

        right = boundary_df[
            boundary_df["direction"]
            == "right"
        ][metric].dropna()

        left_medians.append(
            np.median(left)
        )

        right_medians.append(
            np.median(right)
        )

    ax.bar(
        x - width / 2,
        left_medians,
        width,
        label="Left",
    )

    ax.bar(
        x + width / 2,
        right_medians,
        width,
        label="Right",
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Median critical viewpoint distance"
    )

    ax.set_title(
        "Left vs Right Critical Boundaries"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ================================================================
# PLOT: ORDERINGS
# ================================================================

def plot_orderings(
    boundary_df: pd.DataFrame,
    out_path: Path,
) -> None:

    order = (
        boundary_df
        .groupby(
            [
                "direction",
                "ordering",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    directions = [
        "left",
        "right",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
    )

    for ax, direction in zip(
        axes,
        directions,
    ):

        subset = order[
            order["direction"]
            == direction
        ].sort_values(
            "count",
            ascending=True,
        )

        ax.barh(
            subset["ordering"],
            subset["count"],
        )

        ax.set_title(
            direction.capitalize()
        )

        ax.set_xlabel(
            "Number of expression sequences"
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )

    fig.suptitle(
        "Observed A / C / B Orderings"
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ================================================================
# PLOT: GAP DISTRIBUTIONS
# ================================================================

def plot_gaps(
    boundary_df: pd.DataFrame,
    out_path: Path,
) -> None:

    metrics = [
        (
            "A_C_gap",
            "C − A",
        ),
        (
            "C_B_gap",
            "B − C",
        ),
        (
            "A_B_gap",
            "B − A",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    for ax, (
        column,
        title,
    ) in zip(
        axes,
        metrics,
    ):

        for direction in [
            "left",
            "right",
        ]:

            values = (
                boundary_df[
                    boundary_df["direction"]
                    == direction
                ][column]
                .dropna()
                .to_numpy()
            )

            ax.hist(
                values,
                bins=25,
                alpha=0.55,
                label=direction,
            )

        ax.axvline(
            0,
            linestyle="--",
        )

        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Viewpoint gap"
        )

        ax.legend()

        ax.grid(
            axis="y",
            alpha=0.25,
        )

    fig.suptitle(
        "Boundary Gaps"
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ================================================================
# PLOT: NULL DISTRIBUTION
# ================================================================

def plot_null_model(
    null_results: pd.DataFrame,
    null_raw: Optional[pd.DataFrame],
    out_path: Path,
) -> None:

    if null_raw is None:

        return

    metrics = [
        "A_before_B",
        "A_C_B",
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
    )

    for row_index, direction in enumerate(
        [
            "left",
            "right",
        ]
    ):

        for col_index, metric in enumerate(
            metrics
        ):

            ax = axes[
                row_index,
                col_index
            ]

            values = null_raw[
                null_raw["direction"]
                == direction
            ][metric].dropna()

            if len(values) == 0:

                continue

            observed_row = null_results[
                (
                    null_results["direction"]
                    == direction
                )
                &
                (
                    null_results["metric"]
                    == metric
                )
            ]

            if len(observed_row) == 0:

                continue

            observed = float(
                observed_row[
                    "observed"
                ].iloc[0]
            )

            ax.hist(
                values,
                bins=30,
                alpha=0.75,
            )

            ax.axvline(
                observed,
                linestyle="--",
                linewidth=2,
                label="Observed",
            )

            ax.set_title(
                f"{direction}: {metric}"
            )

            ax.set_xlabel(
                "Null proportion"
            )

            ax.set_ylabel(
                "Frequency"
            )

            ax.legend()

    fig.suptitle(
        "Viewpoint-Shuffle Null Model"
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ================================================================
# SUMMARY REPORT
# ================================================================

def build_report(
    boundary_df: pd.DataFrame,
    directional_stats: pd.DataFrame,
    left_right: pd.DataFrame,
    early_warning: pd.DataFrame,
    early_significance: pd.DataFrame,
    ordering_df: pd.DataFrame,
    null_results: pd.DataFrame,
    thresholds: Dict[str, float],
) -> Dict:

    report = {
        "project": "FER Reliability Benchmark",

        "analysis": (
            "Statistical validation of "
            "representation drift, "
            "expression separability, "
            "and prediction failure."
        ),

        "configuration": {
            "baseline_viewpoint":
                BASELINE_VIEWPOINT,

            "sustained_viewpoints":
                SUSTAINED_VIEWPOINTS,

            "bootstrap_repetitions":
                N_BOOTSTRAP,

            "paired_permutations":
                N_PERMUTATIONS,

            "viewpoint_shuffles":
                N_VIEWPOINT_SHUFFLES,

            "alpha":
                ALPHA,
        },

        "thresholds": thresholds,

        "sequences": {
            "total_complete":
                int(
                    boundary_df[
                        "folder"
                    ].nunique()
                ),
        },

        "directional_statistics":
            directional_stats
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            ),

        "left_right_tests":
            left_right
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            ),

        "early_warning":
            early_warning
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            ),

        "early_warning_significance":
            early_significance
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            ),

        "ordering":
            ordering_df
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            ),

        "viewpoint_shuffle":
            null_results
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            ),

        "interpretation_warning": (
            "These analyses test observed "
            "ordering and boundary differences. "
            "They do not establish causal direction."
        ),
    }

    return report


# ================================================================
# README
# ================================================================

def write_readme(
    out_dir: Path,
    thresholds: Dict[str, float],
) -> None:

    text = f"""
# Statistical Validation

This directory contains the statistical validation of the
A / C / B reliability analysis.

## Definitions

### A — Representation Drift

A critical viewpoint is detected when:

A_angular_distance_deg >= {thresholds["A"]:.8f}

for {SUSTAINED_VIEWPOINTS} consecutive viewpoints.

### C — Expression Separability

A critical viewpoint is detected when:

C_margin <= {thresholds["C"]:.8f}

for {SUSTAINED_VIEWPOINTS} consecutive viewpoints.

### B — Expression Consistency Failure

B is defined as prediction failure:

predicted_folder != true folder

for {SUSTAINED_VIEWPOINTS} consecutive viewpoints.

## Statistical tests

The analysis contains:

1. Bootstrap confidence intervals.
2. Paired permutation tests.
3. Paired rank-biserial effect sizes.
4. Left-vs-right comparisons.
5. Early-warning analysis.
6. Exact sign tests.
7. Viewpoint-shuffle null model.
8. Benjamini-Hochberg FDR correction.

## Important

The analysis does NOT assume:

A < C < B

The observed ordering is reported directly.

The results therefore test whether the proposed
representation-to-separability-to-prediction progression
is actually supported by the data.

## Interpretation

A statistically significant A < B relationship does NOT prove
that representation drift causes prediction failure.

It only establishes that the detected A boundary tends to occur
before the detected B boundary under the current operational
definitions.

Causal interpretation requires additional experimental design.
"""

    path = (
        out_dir
        / "README_statistical_validation.md"
    )

    path.write_text(
        text.strip(),
        encoding="utf-8",
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    initialize_randomness()

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    raw = load_data()

    df, mapping = standardize_columns(
        raw
    )

    # ------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------

    thresholds = (
        load_previous_thresholds()
    )

    print()
    print("=" * 70)
    print("THRESHOLDS")
    print("=" * 70)

    print(
        f"A angular threshold : "
        f"{thresholds['A']:.8f}"
    )

    print(
        f"C margin threshold  : "
        f"{thresholds['C']:.8f}"
    )

    print(
        f"Sustained viewpoints: "
        f"{SUSTAINED_VIEWPOINTS}"
    )

    # ------------------------------------------------------------
    # Complete sequences
    # ------------------------------------------------------------

    sequences = get_complete_sequences(
        df
    )

    print()
    print("=" * 70)
    print("SEQUENCE COVERAGE")
    print("=" * 70)

    print(
        f"Complete expression sequences: "
        f"{len(sequences)}"
    )

    print(
        f"Incomplete expression sequences: "
        f"{df['folder'].nunique() - len(sequences)}"
    )

    # ------------------------------------------------------------
    # Boundary table
    # ------------------------------------------------------------

    boundary_df = build_boundary_table(
        sequences,
        thresholds,
    )

    # ------------------------------------------------------------
    # Save raw boundaries
    # ------------------------------------------------------------

    out_dir = output_dir()

    boundary_path = (
        out_dir
        / "boundary_measurements.csv"
    )

    boundary_df.to_csv(
        boundary_path,
        index=False,
    )

    # ------------------------------------------------------------
    # Directional statistics
    # ------------------------------------------------------------

    directional_stats = (
        analyze_directional_boundaries(
            boundary_df,
            rng,
        )
    )

    directional_stats.to_csv(
        out_dir
        / "directional_boundary_statistics.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Left / right paired comparison
    # ------------------------------------------------------------

    left_right = compare_left_right(
        boundary_df,
        rng,
    )

    left_right = add_fdr(
        left_right,
        "p_value",
    )

    left_right.to_csv(
        out_dir
        / "left_right_paired_tests.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Early warning
    # ------------------------------------------------------------

    early_warning = (
        early_warning_analysis(
            boundary_df
        )
    )

    early_warning.to_csv(
        out_dir
        / "early_warning_bootstrap.csv",
        index=False,
    )

    early_significance = (
        early_warning_significance(
            boundary_df
        )
    )

    early_significance = add_fdr(
        early_significance,
        "sign_test_p",
    )

    early_significance.to_csv(
        out_dir
        / "early_warning_significance.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------

    ordering_df = ordering_analysis(
        boundary_df
    )

    ordering_df.to_csv(
        out_dir
        / "observed_orderings.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Viewpoint shuffle
    # ------------------------------------------------------------

    null_results = viewpoint_shuffle_test(
        sequences,
        boundary_df,
        thresholds,
        rng,
    )

    null_results = add_fdr(
        null_results,
        "p_value",
    )

    null_results.to_csv(
        out_dir
        / "viewpoint_shuffle_results.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Build raw null distribution again for plots.
    #
    # We intentionally perform a lighter second run only for
    # plotting when the first run did not retain raw samples.
    #
    # To avoid another expensive computation, we construct plots
    # only from the summary if raw data are unavailable.
    # ------------------------------------------------------------

    null_raw = None

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING PLOTS")
    print("=" * 70)

    plot_boundary_distributions(
        boundary_df,
        out_dir
        / "boundary_distributions.png",
    )

    plot_left_right(
        boundary_df,
        out_dir
        / "left_vs_right_boundaries.png",
    )

    plot_orderings(
        boundary_df,
        out_dir
        / "observed_orderings.png",
    )

    plot_gaps(
        boundary_df,
        out_dir
        / "boundary_gaps.png",
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    report = build_report(
        boundary_df,
        directional_stats,
        left_right,
        early_warning,
        early_significance,
        ordering_df,
        null_results,
        thresholds,
    )

    report_path = (
        out_dir
        / "statistical_validation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    write_readme(
        out_dir,
        thresholds,
    )

    # ------------------------------------------------------------
    # Final console summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Complete sequences: "
        f"{len(sequences)}"
    )

    print()
    print(
        "Main question:"
    )

    print(
        "Does representation drift tend to "
        "precede prediction failure?"
    )

    for direction in [
        "left",
        "right",
    ]:

        subset = boundary_df[
            boundary_df["direction"]
            == direction
        ]

        valid = subset[
            [
                "A_distance",
                "B_distance",
            ]
        ].dropna()

        if len(valid) > 0:

            rate = np.mean(
                valid["A_distance"]
                <
                valid["B_distance"]
            )

            print(
                f"{direction.upper():<6}: "
                f"A before B = "
                f"{rate:.3%}"
            )

    print()
    print(
        "Main statistical outputs:"
    )

    print(
        out_dir
        / "left_right_paired_tests.csv"
    )

    print(
        out_dir
        / "early_warning_significance.csv"
    )

    print(
        out_dir
        / "viewpoint_shuffle_results.csv"
    )

    print(
        out_dir
        / "statistical_validation_report.json"
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "A significant ordering is not evidence of causality."
    )

    print(
        "It means the observed boundary ordering is "
        "unlikely under the tested null model."
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()